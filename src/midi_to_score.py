"""MIDI → sheet music (MusicXML → PDF + PNG).

Rendering chain (tried in order):
  1. MuseScore CLI      (best quality, requires system install)
  2. LilyPond CLI       (good quality, requires system install)
  3. verovio + svglib   (pure-Python, proper staff notation; pip install verovio svglib reportlab)
  4. matplotlib grand-staff (always available; correct staff lines, note heads, stems, ledger lines)
"""
from __future__ import annotations

import logging
import math
import subprocess
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import find_musescore, find_lilypond, ensure_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ScoreConfig:
    tempo_bpm: float = 120.0
    quantize: bool = True
    quantize_divisions: Tuple[int, ...] = (4, 8)   # 16th-note pass is slow; use 8 by default
    treble_cutoff_midi: int = 60   # C4; notes >= here → treble clef
    title: str = "Transcription"
    dpi: int = 150                 # for PNG export


# ---------------------------------------------------------------------------
# music21 helpers
# ---------------------------------------------------------------------------

def _load_midi_as_score(midi_path: Path, cfg: ScoreConfig):
    """Parse MIDI with music21 and return a cleaned Score."""
    import music21 as m21  # noqa: PLC0415

    logger.info("Parsing MIDI with music21 …")
    score_raw = m21.converter.parse(str(midi_path))

    # ── Set tempo ────────────────────────────────────────────────────────
    from music21 import tempo as m21tempo  # noqa: PLC0415
    mm = m21tempo.MetronomeMark(number=cfg.tempo_bpm)

    # ── Collect all notes into treble / bass lists ───────────────────────
    treble_notes = []
    bass_notes   = []

    for el in score_raw.flat.notesAndRests:
        if hasattr(el, "pitches"):          # Chord
            pitches = el.pitches
        elif hasattr(el, "pitch"):          # Note
            pitches = [el.pitch]
        else:
            continue

        avg_midi = sum(p.midi for p in pitches) / len(pitches)
        if avg_midi >= cfg.treble_cutoff_midi:
            treble_notes.append(el)
        else:
            bass_notes.append(el)

    # ── Build Score with Grand Staff ─────────────────────────────────────
    from music21 import stream, clef, meter, key  # noqa: PLC0415

    def _make_part(notes_list, part_clef, part_id):
        part = stream.Part(id=part_id)
        part.append(mm)
        part.append(meter.TimeSignature("4/4"))
        part.append(part_clef)
        for el in notes_list:
            part.append(el)
        return part

    treble_part = _make_part(treble_notes, clef.TrebleClef(), "Treble")
    bass_part   = _make_part(bass_notes,   clef.BassClef(),   "Bass")

    score = stream.Score([treble_part, bass_part])
    score.metadata = m21.metadata.Metadata()
    score.metadata.title = cfg.title

    # ── Quantise ─────────────────────────────────────────────────────────
    if cfg.quantize:
        logger.info("Quantising score …")
        try:
            score = score.quantize(
                quarterLengthDivisors=cfg.quantize_divisions,
                processOffsets=True,
                processDurations=True,
                inPlace=False,
            )
        except Exception as e:
            logger.warning("Quantisation failed (%s) – using raw timing.", e)

    # Clean up: make accidentals, ties, beams
    try:
        score.makeAccidentals(inPlace=True)
    except Exception:
        pass

    return score


def _score_to_musicxml(score, output_path: Path) -> Path:
    output_path = Path(output_path)
    score.write("musicxml", fp=str(output_path))
    logger.info("MusicXML written: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Renderer: MuseScore
# ---------------------------------------------------------------------------

def _render_musescore(xml_path: Path, output_dir: Path, dpi: int = 150) -> Tuple[Optional[Path], List[Path]]:
    """
    Use MuseScore CLI to produce PDF + PNG pages.
    Returns (pdf_path, [png_paths]).
    """
    mscore = find_musescore()
    if mscore is None:
        return None, []

    ensure_dir(output_dir)
    pdf_path = output_dir / "score.pdf"
    png_stem = output_dir / "score_page"

    logger.info("Rendering with MuseScore (%s) …", mscore)

    # PDF
    r = subprocess.run(
        [mscore, "-o", str(pdf_path), str(xml_path)],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        logger.warning("MuseScore PDF failed: %s", r.stderr[-500:])
        pdf_path = None

    # PNG (MuseScore appends -1.png, -2.png, …)
    r2 = subprocess.run(
        [mscore, "-o", str(png_stem) + ".png", "--image-resolution", str(dpi), str(xml_path)],
        capture_output=True, text=True, timeout=120,
    )
    png_paths: List[Path] = []
    if r2.returncode == 0:
        # MuseScore 3 names: score_page-1.png; MuseScore 4 names: score_page-1.png
        for suffix in ["-1", "-2", "-3", "-4", "-5", "-6", "-7", "-8"]:
            p = output_dir / f"score_page{suffix}.png"
            if p.exists():
                png_paths.append(p)
        if not png_paths:
            # Maybe it wrote score_page.png (single page)
            sp = output_dir / "score_page.png"
            if sp.exists():
                png_paths.append(sp)
    else:
        logger.warning("MuseScore PNG failed: %s", r2.stderr[-500:])

    return pdf_path if pdf_path and pdf_path.exists() else None, png_paths


# ---------------------------------------------------------------------------
# Renderer: LilyPond
# ---------------------------------------------------------------------------

def _render_lilypond(score, output_dir: Path) -> Tuple[Optional[Path], List[Path]]:
    """Convert music21 score → LilyPond → PDF + PNG."""
    lilypond = find_lilypond()
    if lilypond is None:
        return None, []

    ensure_dir(output_dir)
    ly_path = output_dir / "score.ly"

    logger.info("Rendering with LilyPond …")
    try:
        score.write("lily", fp=str(ly_path))
    except Exception as e:
        logger.warning("music21 LilyPond export failed: %s", e)
        return None, []

    r = subprocess.run(
        [lilypond, "--png", f"--output={output_dir / 'score'}", str(ly_path)],
        capture_output=True, text=True, timeout=120,
    )
    pdf_path = output_dir / "score.pdf"
    png_paths = sorted(output_dir.glob("score*.png"))
    return (pdf_path if pdf_path.exists() else None), png_paths


# ---------------------------------------------------------------------------
# Renderer: verovio (MusicXML → SVG → PNG)  – pure-Python, proper notation
# ---------------------------------------------------------------------------

def _svg_to_png(svg_path: Path, png_path: Path, dpi: int) -> bool:
    """
    Convert an SVG file to PNG.  Tries (in order):
      1. svglib + reportlab  (pure Python)
      2. ImageMagick `magick` / `convert` CLI
    Returns True on success.
    """
    # ── Try svglib ────────────────────────────────────────────────────────
    try:
        from svglib.svglib import svg2rlg       # noqa: PLC0415
        from reportlab.graphics import renderPM  # noqa: PLC0415
        drawing = svg2rlg(str(svg_path))
        if drawing is not None:
            renderPM.drawToFile(drawing, str(png_path), fmt="PNG", dpi=dpi)
            if png_path.exists() and png_path.stat().st_size > 0:
                return True
    except Exception as exc:
        logger.debug("svglib failed: %s", exc)

    # ── Try ImageMagick CLI ───────────────────────────────────────────────
    for cmd_name in ("magick", "convert"):
        magick = shutil.which(cmd_name)
        if magick is None:
            continue
        r = subprocess.run(
            [magick, "-density", str(dpi), str(svg_path), str(png_path)],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0 and png_path.exists() and png_path.stat().st_size > 0:
            return True

    return False


def _render_verovio(
    xml_path: Path, output_dir: Path, cfg: ScoreConfig
) -> Tuple[None, List[Path]]:
    """
    Convert MusicXML → SVG pages with verovio, then SVG → PNG.

    SVG conversion strategy (tried in order):
      1. svglib + reportlab  (pure Python)
      2. ImageMagick `magick`/`convert` CLI

    Returns (None, [png_paths]).
    """
    try:
        import verovio  # noqa: PLC0415
    except ImportError:
        logger.debug("verovio not installed; skipping (pip install verovio importlib_resources)")
        return None, []

    ensure_dir(output_dir)
    logger.info("Rendering with verovio …")

    try:
        tk = verovio.toolkit()
        tk.setOptions({
            "adjustPageHeight": "true",
            "footer":           "none",
            "header":           "none",
        })
        xml_data = xml_path.read_text(encoding="utf-8")
        if not tk.loadData(xml_data):
            logger.warning("verovio could not load MusicXML")
            return None, []

        page_count = tk.getPageCount()
        png_paths: List[Path] = []

        for page_num in range(1, page_count + 1):
            svg_str  = tk.renderToSVG(page_num)
            svg_tmp  = output_dir / f"_verovio_p{page_num}.svg"
            png_path = output_dir / f"score_page-{page_num}.png"
            svg_tmp.write_text(svg_str, encoding="utf-8")
            try:
                if _svg_to_png(svg_tmp, png_path, cfg.dpi):
                    png_paths.append(png_path)
                else:
                    logger.warning("SVG→PNG conversion failed for page %d", page_num)
            finally:
                svg_tmp.unlink(missing_ok=True)

        if png_paths:
            logger.info("verovio rendered %d page(s)", len(png_paths))
        return None, png_paths

    except Exception as exc:
        logger.warning("verovio rendering failed: %s", exc)
        return None, []


# ---------------------------------------------------------------------------
# Renderer: matplotlib grand-staff (always available last-resort fallback)
# ---------------------------------------------------------------------------

# Chromatic pitch class → (diatonic_step_in_octave, accidental: 0=natural 1=sharp)
_PC_TO_DIAT: Dict[int, Tuple[int, int]] = {
    0:  (0, 0),   # C
    1:  (0, 1),   # C#
    2:  (1, 0),   # D
    3:  (1, 1),   # D#
    4:  (2, 0),   # E
    5:  (3, 0),   # F
    6:  (3, 1),   # F#
    7:  (4, 0),   # G
    8:  (4, 1),   # G#
    9:  (5, 0),   # A
    10: (5, 1),   # A# / Bb
    11: (6, 0),   # B
}

# Reference diatonic totals for bottom lines of each staff:
#   Treble bottom = E4 (MIDI 64): raw_oct=5, diat_in_oct=2 → 5*7+2 = 37
#   Bass   bottom = G2 (MIDI 43): raw_oct=3, diat_in_oct=4 → 3*7+4 = 25
_TREBLE_DIAT_REF = 37
_BASS_DIAT_REF   = 25


def _midi_staff_pos(pitch: int, treble: bool) -> Tuple[float, int]:
    """
    Map a MIDI pitch to a staff position and accidental symbol.

    Returns
    -------
    (staff_pos, accidental)
      staff_pos  : float – 0.0 = bottom line, 0.5 = first space, …, 4.0 = top line.
      accidental : int   – 0 = none, 1 = sharp (♯).
    """
    raw_oct = pitch // 12
    pc      = pitch % 12
    diat_in_oct, acc = _PC_TO_DIAT[pc]
    total   = raw_oct * 7 + diat_in_oct
    ref     = _TREBLE_DIAT_REF if treble else _BASS_DIAT_REF
    return (total - ref) * 0.5, acc


def _render_staff_matplotlib(
    midi_path: Path,
    output_dir: Path,
    cfg: ScoreConfig,
) -> Tuple[None, List[Path]]:
    """
    Draw a proper treble+bass grand staff using matplotlib.

    Produces real staff notation: 5-line staves, note heads at correct staff
    positions, stems, ledger lines, and ♯ accidentals.
    """
    import pretty_midi              # noqa: PLC0415
    import matplotlib               # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt # noqa: PLC0415
    from matplotlib.patches import Ellipse  # noqa: PLC0415

    ensure_dir(output_dir)
    png_path = output_dir / "score_page-1.png"

    # ── Load notes ─────────────────────────────────────────────────────────
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    all_notes = sorted(
        [n for inst in pm.instruments if not inst.is_drum for n in inst.notes],
        key=lambda n: n.start,
    )
    total_dur = max((n.end for n in all_notes), default=4.0)

    # ── Layout (data coordinates) ───────────────────────────────────────────
    # Staff line spacing = STAFF_SP data units.
    # Treble bottom line (E4) at y = TREBLE_Y0
    # Bass   bottom line (G2) at y = BASS_Y0
    STAFF_SP  = 1.0
    TREBLE_Y0 = 9.0   # treble lines: 9, 10, 11, 12, 13
    BASS_Y0   = 0.0   # bass   lines:  0,  1,  2,  3,  4
    # Gap [4 … 9] holds D4 – D5 transition with middle C at 8 (treble: sp=-1)
    # and at 5 (bass: sp=5).

    MARGIN_L  = 3.5   # room for clef symbol
    MARGIN_R  = 1.0
    X_SCALE   = 25.0 / max(total_dur, 1.0)   # data-units per second
    x_total   = MARGIN_L + total_dur * X_SCALE + MARGIN_R

    Y_MIN = BASS_Y0   - 3.5   # space for low ledger lines
    Y_MAX = TREBLE_Y0 + 4.0 + 3.5  # space for high ledger lines + title
    FIG_W, FIG_H = 16.0, 5.5  # inches

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor="white")
    ax.set_facecolor("white")

    # ── Staff lines ─────────────────────────────────────────────────────────
    line_kw = dict(color="black", lw=1.1, zorder=1, solid_capstyle="round")
    for i in range(5):
        ax.plot([0, x_total], [TREBLE_Y0 + i, TREBLE_Y0 + i], **line_kw)
        ax.plot([0, x_total], [BASS_Y0   + i, BASS_Y0   + i], **line_kw)

    # ── Grand-staff bracket & terminal bar lines ─────────────────────────────
    bracket_bot = BASS_Y0
    bracket_top = TREBLE_Y0 + 4 * STAFF_SP
    ax.plot([0, 0],       [bracket_bot, bracket_top], color="black", lw=3.5, zorder=2)
    ax.plot([x_total, x_total], [bracket_bot, bracket_top], color="black", lw=1.5, zorder=2)

    # ── Clef symbols ────────────────────────────────────────────────────────
    # Draw simplified clef shapes using matplotlib primitives (no special fonts).
    # Treble clef: oval at G4 line (sp=1) with vertical tail
    # G4 is on the second line from bottom of treble staff
    g4_y = TREBLE_Y0 + 1.0 * STAFF_SP   # G4 line
    from matplotlib.patches import Ellipse as _Ellipse, Arc as _Arc  # noqa: PLC0415
    # Oval (the circle that wraps around G4 line)
    oval = _Ellipse((1.0, g4_y), width=0.55, height=1.6,
                    facecolor="white", edgecolor="black", lw=1.6, zorder=3)
    ax.add_patch(oval)
    # Vertical line through the oval (the stem of the treble clef)
    ax.plot([1.0, 1.0], [TREBLE_Y0 - 1.5, TREBLE_Y0 + 4.5 * STAFF_SP],
            color="black", lw=1.6, zorder=2, solid_capstyle="round")
    # Curl at the top
    curl = _Arc((1.0, TREBLE_Y0 + 4.5 * STAFF_SP), width=0.8, height=0.8,
                angle=0, theta1=0, theta2=270, color="black", lw=1.4, zorder=3)
    ax.add_patch(curl)
    # Short tail at bottom
    ax.plot([0.75, 1.25], [TREBLE_Y0 - 1.5, TREBLE_Y0 - 1.5],
            color="black", lw=1.4, zorder=3, solid_capstyle="round")

    # Bass clef: "F" with two dots at F3 line (sp=3)
    f3_y = BASS_Y0 + 3.0 * STAFF_SP   # F3 = 4th bass line
    ax.plot([0.65, 1.3], [f3_y + 0.4, f3_y + 0.4],
            color="black", lw=1.6, zorder=3, solid_capstyle="round")
    ax.plot([0.65, 0.65], [f3_y - 0.8, f3_y + 0.4],
            color="black", lw=1.6, zorder=3, solid_capstyle="round")
    ax.plot([0.65, 1.2], [f3_y - 0.2, f3_y - 0.2],
            color="black", lw=1.4, zorder=3, solid_capstyle="round")
    # Two dots to the right of the clef stem (between 4th and 5th bass lines)
    for dot_y in (f3_y + 0.75, f3_y + 0.25):
        dot = _Ellipse((1.55, dot_y), width=0.18, height=0.28,
                       facecolor="black", edgecolor="none", zorder=3)
        ax.add_patch(dot)

    # ── Note-head sizing (keep circular in figure pixels) ───────────────────
    y_data_range = Y_MAX - Y_MIN
    ipdu_y = FIG_H / y_data_range          # inches per data-unit (y)
    ipdu_x = FIG_W / max(x_total, 1.0)    # inches per data-unit (x)
    NOTE_RY  = 0.42 * STAFF_SP
    NOTE_RX  = NOTE_RY * (ipdu_y / ipdu_x)
    STEM_LEN = 3.0 * STAFF_SP
    LW_LEDGER = 1.1

    # ── Draw notes ──────────────────────────────────────────────────────────
    for note in all_notes:
        is_treble = note.pitch >= cfg.treble_cutoff_midi
        sp, acc   = _midi_staff_pos(note.pitch, is_treble)
        y0_staff  = TREBLE_Y0 if is_treble else BASS_Y0
        x_note    = MARGIN_L + note.start * X_SCALE
        y_note    = y0_staff + sp * STAFF_SP

        # Note head (tilted ellipse, like real note heads)
        el = Ellipse(
            (x_note, y_note),
            width=2 * NOTE_RX, height=2 * NOTE_RY,
            angle=-18,
            facecolor="black", edgecolor="black", zorder=4,
        )
        ax.add_patch(el)

        # Stem: up when note is in lower half of staff, down otherwise
        staff_mid = y0_staff + 2.0 * STAFF_SP
        stem_up   = y_note <= staff_mid
        if stem_up:
            ax.plot([x_note + NOTE_RX * 0.75, x_note + NOTE_RX * 0.75],
                    [y_note, y_note + STEM_LEN],
                    color="black", lw=1.0, zorder=4)
        else:
            ax.plot([x_note - NOTE_RX * 0.75, x_note - NOTE_RX * 0.75],
                    [y_note, y_note - STEM_LEN],
                    color="black", lw=1.0, zorder=4)

        # Ledger lines below staff (integer positions ≤ -1)
        if sp < -0.25:
            lowest = math.ceil(sp)  # highest integer at-or-below sp (e.g. sp=-1.5 → -1)
            for lpos in range(-1, lowest - 1, -1):
                y_led = y0_staff + lpos * STAFF_SP
                ax.plot([x_note - NOTE_RX * 1.9, x_note + NOTE_RX * 1.9],
                        [y_led, y_led], color="black", lw=LW_LEDGER, zorder=3)

        # Ledger lines above staff (integer positions ≥ 5)
        if sp > 4.25:
            highest = math.floor(sp)  # lowest integer at-or-above sp
            for lpos in range(5, highest + 1):
                y_led = y0_staff + lpos * STAFF_SP
                ax.plot([x_note - NOTE_RX * 1.9, x_note + NOTE_RX * 1.9],
                        [y_led, y_led], color="black", lw=LW_LEDGER, zorder=3)

        # Accidental (sharp only; flats are not generated by _PC_TO_DIAT)
        if acc:
            ax.text(x_note - NOTE_RX - 0.25, y_note, "♯",
                    fontsize=9, va="center", ha="right",
                    color="black", zorder=5)

    # ── Title ──────────────────────────────────────────────────────────────
    ax.set_title(cfg.title, fontsize=12, color="black", pad=8)

    # ── Axes ───────────────────────────────────────────────────────────────
    ax.set_xlim(-0.5, x_total + 0.5)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.axis("off")

    plt.tight_layout(pad=0.3)
    fig.savefig(str(png_path), dpi=cfg.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Grand-staff notation (matplotlib) saved: %s", png_path)
    return None, [png_path]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def midi_to_score(
    midi_path: Path,
    output_dir: Path,
    cfg: Optional[ScoreConfig] = None,
) -> Tuple[Optional[Path], List[Path], Optional[Path]]:
    """
    Convert MIDI to sheet music.

    Returns
    -------
    (pdf_path, png_paths, xml_path)
      pdf_path  – Path to PDF or None
      png_paths – list of PNG page paths (may be piano-roll fallback)
      xml_path  – MusicXML path or None
    """
    if cfg is None:
        cfg = ScoreConfig()

    ensure_dir(output_dir)

    # Check cache
    xml_path = output_dir / "score.xml"
    pdf_path_cached = output_dir / "score.pdf"
    existing_pngs = sorted(output_dir.glob("score_page*.png"))
    if existing_pngs and xml_path.exists():
        logger.info("Using cached score files.")
        return (
            pdf_path_cached if pdf_path_cached.exists() else None,
            existing_pngs,
            xml_path,
        )

    # ── Parse MIDI → music21 Score ────────────────────────────────────────
    score = None
    try:
        score = _load_midi_as_score(midi_path, cfg)
        xml_path = _score_to_musicxml(score, output_dir / "score.xml")
    except Exception as e:
        logger.error("music21 failed: %s – falling back to staff renderer", e)
        _, png_paths = _render_staff_matplotlib(midi_path, output_dir, cfg)
        return None, png_paths, None

    # ── Try MuseScore ─────────────────────────────────────────────────────
    pdf_path, png_paths = _render_musescore(xml_path, output_dir, dpi=cfg.dpi)
    if png_paths:
        logger.info("Sheet music via MuseScore: %d page(s)", len(png_paths))
        return pdf_path, png_paths, xml_path

    # ── Try LilyPond ──────────────────────────────────────────────────────
    pdf_path, png_paths = _render_lilypond(score, output_dir)
    if png_paths:
        logger.info("Sheet music via LilyPond: %d page(s)", len(png_paths))
        return pdf_path, png_paths, xml_path

    # ── Try verovio (pure-Python, proper staff notation) ─────────────────
    _, png_paths = _render_verovio(xml_path, output_dir, cfg)
    if png_paths:
        logger.info("Sheet music via verovio: %d page(s)", len(png_paths))
        return None, png_paths, xml_path

    # ── matplotlib grand-staff (always available) ─────────────────────────
    logger.warning(
        "Neither MuseScore, LilyPond, nor verovio produced output. "
        "Using matplotlib grand-staff renderer."
    )
    _, png_paths = _render_staff_matplotlib(midi_path, output_dir, cfg)
    return None, png_paths, xml_path
