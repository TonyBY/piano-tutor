"""Render a "falling notes" piano tutorial video (MP4).

Layout (top → bottom)
─────────────────────
┌────────────────────────────────────┐
│  Sheet-music / piano-roll overlay  │  sheet_height px
├────────────────────────────────────┤
│                                    │
│      Falling note bars             │  dynamic area
│                                    │
├────────────────────────────────────┤
│  Piano keyboard  (labeled keys)    │  keyboard_height px
└────────────────────────────────────┘

Frame pipeline:  numpy  ──▶  OpenCV drawing  ──▶  ffmpeg stdin pipe  ──▶  H.264 MP4
Audio mux:       silent MP4 + WAV  ──▶  ffmpeg  ──▶  final MP4
"""
from __future__ import annotations

import bisect
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None   # type: ignore[assignment]
    np = None    # type: ignore[assignment]
    _CV2_AVAILABLE = False


def _require_cv2() -> None:
    """Raise a descriptive error if OpenCV or NumPy is not installed."""
    if not _CV2_AVAILABLE:
        raise ImportError(
            "opencv-python-headless and numpy are required for video rendering.\n"
            "Install with:  pip install opencv-python-headless numpy"
        )

from .utils import BLACK_KEY_CLASSES, NOTE_NAMES_SHARP, PIANO_MIN_MIDI, PIANO_MAX_MIDI, ensure_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VideoConfig:
    video_width: int   = 1280
    video_height: int  = 720
    fps: int           = 30
    fall_speed: float  = 200.0         # px / second
    key_range_min: int = PIANO_MIN_MIDI
    key_range_max: int = PIANO_MAX_MIDI
    keyboard_height: int = 140
    sheet_height: int    = 175
    # colours (BGR for OpenCV)
    bg_color: Tuple[int, int, int]       = (18, 18, 30)
    divider_color: Tuple[int, int, int]  = (50, 50, 72)


# ---------------------------------------------------------------------------
# Pitch → colour  (one warm colour per pitch class, cycling octave brightness)
# ---------------------------------------------------------------------------

# HSV-inspired colours for each of the 12 pitch classes, in BGR
_PITCH_BGR = [
    (50,  80, 230),   # C
    (50, 130, 240),   # C#
    (50, 200, 230),   # D
    (40, 220, 170),   # D#
    (40, 220,  90),   # E
    (80, 220,  40),   # F
    (160, 220,  40),  # F#
    (220, 190,  40),  # G
    (220, 130,  40),  # G#
    (220,  60,  60),  # A
    (190,  50, 200),  # A#
    (110,  50, 220),  # B
]


def _note_color(midi: int) -> Tuple[int, int, int]:
    base = _PITCH_BGR[midi % 12]
    oct_ = midi // 12          # 1..9 for full piano range
    factor = 0.6 + 0.04 * oct_
    return tuple(int(min(255, c * factor)) for c in base)


# ---------------------------------------------------------------------------
# Piano keyboard layout
# ---------------------------------------------------------------------------

def build_key_layout(video_width: int, key_range: Tuple[int, int]) -> List[dict]:
    """
    Return a list of dicts (one per MIDI note in range), each with:
      { note, x, w, is_black, name, color }
    Sorted lowest → highest pitch.
    """
    lo, hi = key_range
    notes = list(range(lo, hi + 1))

    white_notes = [n for n in notes if n % 12 not in BLACK_KEY_CLASSES]
    n_white = max(1, len(white_notes))

    white_w = video_width / n_white
    black_w = white_w * 0.58

    # Pre-map white note → x
    white_x: dict[int, float] = {
        n: i * white_w for i, n in enumerate(white_notes)
    }

    keys = []
    for note in notes:
        nclass = note % 12
        is_black = nclass in BLACK_KEY_CLASSES
        name = NOTE_NAMES_SHARP[nclass]
        color = _note_color(note)

        if not is_black:
            x = white_x[note]
            keys.append(dict(note=note, x=x, w=white_w, is_black=False,
                             name=name, color=color))
        else:
            # Place black key centred on the boundary between two white keys
            prev_w = note - 1
            while prev_w >= lo and prev_w % 12 in BLACK_KEY_CLASSES:
                prev_w -= 1
            px = white_x.get(prev_w, 0.0) + white_w
            x = px - black_w / 2
            keys.append(dict(note=note, x=x, w=black_w, is_black=True,
                             name=name, color=color))

    return keys


# ---------------------------------------------------------------------------
# Static keyboard image
# ---------------------------------------------------------------------------

def render_keyboard_image(key_layout: List[dict], cfg: VideoConfig):
    """Return (keyboard_height × video_width × 3) uint8 BGR image."""
    _require_cv2()
    h, w = cfg.keyboard_height, cfg.video_width
    img = np.full((h, w, 3), 242, dtype=np.uint8)  # off-white

    bh = int(h * 0.62)  # black-key height in pixels

    # White keys
    for k in key_layout:
        if k["is_black"]:
            continue
        x1 = int(k["x"]) + 1
        x2 = int(k["x"] + k["w"]) - 1
        cv2.rectangle(img, (x1, 0), (x2, h - 1), (248, 248, 248), -1)
        cv2.rectangle(img, (x1, 0), (x2, h - 1), (60, 60, 60), 1)

        # Label C notes
        if k["note"] % 12 == 0:
            oct_ = k["note"] // 12 - 1
            label = f"C{oct_}"
            font_scale = 0.32
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            tx = x1 + (x2 - x1 - tw) // 2
            ty = h - 7
            cv2.putText(img, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (90, 90, 90), 1, cv2.LINE_AA)

    # Black keys
    for k in key_layout:
        if not k["is_black"]:
            continue
        x1 = int(k["x"]) + 1
        x2 = int(k["x"] + k["w"]) - 1
        cv2.rectangle(img, (x1, 0), (x2, bh), (28, 28, 28), -1)
        cv2.rectangle(img, (x1 + 1, 0), (x2 - 1, 5), (60, 60, 60), -1)  # top sheen

    return img


def highlight_active_keys(
    base_img,
    key_layout: List[dict],
    active: Set[int],
    cfg: VideoConfig,
):
    """Return copy of *base_img* with active keys glowing."""
    _require_cv2()
    if not active:
        return base_img
    img = base_img.copy()
    h = cfg.keyboard_height
    bh = int(h * 0.62)

    # White keys first (under black keys)
    for k in key_layout:
        if k["is_black"] or k["note"] not in active:
            continue
        x1 = int(k["x"]) + 1
        x2 = int(k["x"] + k["w"]) - 1
        c = k["color"]
        top_y = h - int(h * 0.55)
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, top_y), (x2, h - 2), c, -1)
        cv2.addWeighted(img, 0.25, overlay, 0.75, 0, img)
        cv2.rectangle(img, (x1, 0), (x2, h - 1), (60, 60, 60), 1)

    # Black keys on top
    for k in key_layout:
        if not k["is_black"] or k["note"] not in active:
            continue
        x1 = int(k["x"]) + 1
        x2 = int(k["x"] + k["w"]) - 1
        c = k["color"]
        # Glow at the bottom of the black key
        glow_top = max(0, bh - 25)
        cv2.rectangle(img, (x1, glow_top), (x2, bh), c, -1)
        # Keep the rest dark
        cv2.rectangle(img, (x1, 0), (x2, glow_top), (40, 40, 40), -1)
        cv2.rectangle(img, (x1 + 1, 0), (x2 - 1, 4), (70, 70, 70), -1)

    return img


# ---------------------------------------------------------------------------
# Score strip (for top overlay)
# ---------------------------------------------------------------------------

def prepare_score_strip(
    png_paths: List[Path],
    strip_height: int,
    video_width: int,
) -> Optional[object]:
    """
    Load score pages, resize to *strip_height*, concatenate horizontally.
    Returns BGR ndarray or None.
    """
    _require_cv2()
    pages = []
    for p in png_paths:
        if not Path(p).exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        oh, ow = img.shape[:2]
        new_w = max(1, int(ow * strip_height / oh))
        pages.append(cv2.resize(img, (new_w, strip_height), interpolation=cv2.INTER_AREA))

    if not pages:
        return None
    return np.concatenate(pages, axis=1)


# ---------------------------------------------------------------------------
# MIDI note events
# ---------------------------------------------------------------------------

def load_note_events(midi_path: Path) -> Tuple[List[Tuple], float]:
    """
    Returns (notes, total_duration_sec).
    notes: sorted list of (start_sec, end_sec, midi_pitch, velocity)
    """
    import pretty_midi  # noqa: PLC0415
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    dur = pm.get_end_time()
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append((float(n.start), float(n.end), int(n.pitch), int(n.velocity)))
    notes.sort()
    logger.info("Loaded %d notes, duration %.1fs", len(notes), dur)
    return notes, dur


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def _render_score_overlay(
    frame: np.ndarray,
    strip: Optional[np.ndarray],
    current_t: float,
    total_dur: float,
    cfg: VideoConfig,
) -> None:
    sh = cfg.sheet_height
    vw = cfg.video_width
    frame[:sh, :] = (242, 242, 248)  # light background

    if strip is None or total_dur <= 0:
        msg = "Install MuseScore or LilyPond for sheet-music overlay"
        cv2.putText(frame, msg, (10, sh // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 130), 1, cv2.LINE_AA)
        return

    sw = strip.shape[1]
    max_scroll = max(0, sw - vw)
    sx = int(max_scroll * min(1.0, current_t / total_dur)) if total_dur > 0 else 0
    crop = strip[:, sx: sx + vw]
    if crop.shape[1] < vw:
        pad = np.full((sh, vw - crop.shape[1], 3), 242, dtype=np.uint8)
        crop = np.concatenate([crop, pad], axis=1)
    frame[:sh, :] = crop[:sh, :]

    # Playhead tick
    px = int(vw * min(1.0, current_t / total_dur))
    cv2.line(frame, (px, sh - 4), (px, sh), (0, 100, 255), 2)


def _render_frame(
    frame: np.ndarray,
    t: float,
    total_dur: float,
    notes: List[Tuple],
    note_map: dict,      # midi_pitch -> key dict
    start_times: list,   # sorted list of start times for bisect
    keyboard_base: np.ndarray,
    kb_cache: list,      # [cached_img, frozenset(active_pitches)] – mutated in-place
    strip: Optional[np.ndarray],
    cfg: VideoConfig,
) -> None:
    sh = cfg.sheet_height
    kh = cfg.keyboard_height
    vw = cfg.video_width
    vh = cfg.video_height
    keyboard_top = vh - kh
    notes_top    = sh + 2
    notes_bottom = keyboard_top
    notes_height = notes_bottom - notes_top

    # ── Background ────────────────────────────────────────────────────────
    frame[:] = cfg.bg_color

    # ── Score overlay ─────────────────────────────────────────────────────
    _render_score_overlay(frame, strip, t, total_dur, cfg)

    # ── Dividers ──────────────────────────────────────────────────────────
    cv2.line(frame, (0, sh + 1),        (vw, sh + 1),        cfg.divider_color, 2)
    cv2.line(frame, (0, keyboard_top),  (vw, keyboard_top),  cfg.divider_color, 2)

    # ── Falling note bars ─────────────────────────────────────────────────
    lookahead = notes_height / cfg.fall_speed   # seconds of look-ahead visible
    active_pitches: Set[int] = set()

    # Binary search: only iterate notes that could be visible
    earliest = t - 0.1             # notes currently playing
    latest   = t + lookahead + 0.5
    lo_idx   = max(0, bisect.bisect_left(start_times, earliest) - 10)

    for idx in range(lo_idx, len(notes)):
        ts, te, pitch, vel = notes[idx]
        if ts > latest:
            break
        if te < t - 0.05:
            continue
        if pitch not in note_map:
            continue

        k = note_map[pitch]
        color = k["color"]
        kx, kw = k["x"], k["w"]

        # y positions (keyboard_top ↔ current_time)
        y_bot_f = keyboard_top - (ts - t) * cfg.fall_speed
        y_top_f = keyboard_top - (te - t) * cfg.fall_speed

        y_bot = int(min(keyboard_top - 1, y_bot_f))
        y_top = int(max(notes_top, y_top_f))

        if y_bot <= notes_top:
            continue
        if y_top >= keyboard_top:
            continue
        if y_bot - y_top < 2:
            if ts <= t <= te:
                y_top = keyboard_top - 5
                y_bot = keyboard_top - 1
            else:
                continue

        x1 = int(kx) + 1
        x2 = max(x1 + 2, int(kx + kw) - 1)

        # Glow halo (wider, darker)
        gx1 = max(0, x1 - 4)
        gx2 = min(vw - 1, x2 + 4)
        dim = tuple(c // 3 for c in color)
        cv2.rectangle(frame, (gx1, y_top), (gx2, y_bot), dim, -1)

        # Main bar
        cv2.rectangle(frame, (x1, y_top), (x2, y_bot), color, -1)

        # Top highlight
        hi = tuple(min(255, c + 90) for c in color)
        cv2.rectangle(frame, (x1, y_top), (x2, min(y_bot, y_top + 3)), hi, -1)

        if ts <= t <= te:
            active_pitches.add(pitch)

    # ── Piano keyboard (cache: only re-render when active keys change) ────────
    frozen = frozenset(active_pitches)
    if frozen != kb_cache[1]:
        kb_cache[0] = highlight_active_keys(
            keyboard_base,
            [note_map[p] for p in active_pitches if p in note_map],
            active_pitches, cfg,
        )
        kb_cache[1] = frozen
    frame[keyboard_top:, :] = kb_cache[0]

    # ── HUD: time counter ─────────────────────────────────────────────────
    hud = f"{max(0.0, t):.1f} / {total_dur:.1f} s"
    cv2.putText(frame, hud, (8, sh + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 190), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main render entry point
# ---------------------------------------------------------------------------

def render_falling_notes_video(
    midi_path: Path,
    audio_path: Path,
    score_png_paths: List[Path],
    output_path: Path,
    cfg: Optional[VideoConfig] = None,
    progress_cb=None,       # callable(fraction: float)
) -> Path:
    """
    Render the tutorial video and return path to final MP4.

    Parameters
    ----------
    midi_path       : source MIDI
    audio_path      : WAV to mux into the video
    score_png_paths : list of score-page PNGs for overlay
    output_path     : desired output MP4 path
    cfg             : VideoConfig (defaults used if None)
    progress_cb     : optional callback with float 0..1
    """
    _require_cv2()

    if cfg is None:
        cfg = VideoConfig()

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    # ── Load MIDI ─────────────────────────────────────────────────────────
    notes, total_dur = load_note_events(midi_path)
    if total_dur < 0.5:
        raise ValueError("MIDI appears to have no note content.")

    start_times = [n[0] for n in notes]

    # ── Key layout ────────────────────────────────────────────────────────
    key_range = (cfg.key_range_min, cfg.key_range_max)
    key_layout = build_key_layout(cfg.video_width, key_range)
    note_map   = {k["note"]: k for k in key_layout}

    # ── Static keyboard image ─────────────────────────────────────────────
    keyboard_base = render_keyboard_image(key_layout, cfg)

    # ── Score strip ───────────────────────────────────────────────────────
    strip = prepare_score_strip(score_png_paths, cfg.sheet_height, cfg.video_width)

    # ── ffmpeg process ────────────────────────────────────────────────────
    notes_area_h = cfg.video_height - cfg.keyboard_height - cfg.sheet_height - 4
    lookahead_s  = notes_area_h / cfg.fall_speed
    buffer_pre   = lookahead_s + 0.5   # seconds before beat 1
    buffer_post  = 2.0                 # trailing seconds

    total_video_dur = total_dur + buffer_pre + buffer_post
    n_frames        = int(total_video_dur * cfg.fps) + 1

    temp_mp4 = output_path.with_name(output_path.stem + "__tmp" + output_path.suffix)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{cfg.video_width}x{cfg.video_height}",
        "-r", str(cfg.fps),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-crf", "22",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(temp_mp4),
    ]

    logger.info(
        "Rendering %d frames (%.1fs @ %dfps) …", n_frames, total_video_dur, cfg.fps
    )
    frame = np.zeros((cfg.video_height, cfg.video_width, 3), dtype=np.uint8)

    proc = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        report_every = max(1, n_frames // 200)
        # [current_keyboard_img, frozenset_of_active_pitches] – mutated per-frame
        kb_cache = [keyboard_base, frozenset()]
        for fi in range(n_frames):
            t = fi / cfg.fps - buffer_pre
            _render_frame(
                frame, t, total_dur, notes, note_map,
                start_times, keyboard_base, kb_cache, strip, cfg,
            )
            proc.stdin.write(frame.tobytes())
            if progress_cb and fi % report_every == 0:
                progress_cb(fi / n_frames)

        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg video encoding failed.")
    except Exception:
        proc.kill()
        if temp_mp4.exists():
            temp_mp4.unlink(missing_ok=True)
        raise

    # ── Mux audio ─────────────────────────────────────────────────────────
    logger.info("Muxing audio …")
    _mux_audio(temp_mp4, audio_path, output_path, buffer_pre)

    if temp_mp4.exists():
        temp_mp4.unlink(missing_ok=True)

    if progress_cb:
        progress_cb(1.0)

    logger.info("Video ready: %s", output_path)
    return output_path


def _mux_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    audio_delay: float = 0.0,
) -> None:
    """Combine video track with audio, adding *audio_delay* seconds of padding."""
    # Build audio filter: add silence at the start to align with buffer_pre
    af = f"adelay={int(audio_delay * 1000)}:all=1,apad"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-af", af,
        "-shortest",
        str(output_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.warning("Audio mux failed (%s). Copying video-only file.", r.stderr[-300:])
        import shutil
        shutil.copy(str(video_path), str(output_path))
