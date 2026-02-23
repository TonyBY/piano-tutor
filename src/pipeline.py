"""Pipeline orchestrator: glues audio_io → transcribe → midi_to_score → render_video."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .audio_io import prepare_audio
from .transcribe import transcribe
from .midi_to_score import midi_to_score, ScoreConfig
from .render_video import render_falling_notes_video, VideoConfig
from .utils import cache_subdir, ensure_dir

logger = logging.getLogger(__name__)

CACHE_ROOT = Path("cache")


# ---------------------------------------------------------------------------
# Config / Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # Audio
    sample_rate: int = 22050
    max_duration: Optional[float] = 120.0    # seconds; None = no limit

    # Transcription
    transcription_mode: str = "polyphonic"   # 'polyphonic' | 'monophonic'
    onset_threshold: float  = 0.5
    frame_threshold: float  = 0.3
    min_note_ms: float      = 58.0

    # Score
    quantize: bool = True
    score: ScoreConfig = field(default_factory=ScoreConfig)

    # Video
    video: VideoConfig = field(default_factory=VideoConfig)

    # Cache
    cache_root: Path = CACHE_ROOT


@dataclass
class PipelineResult:
    wav_path:   Optional[Path] = None
    midi_path:  Optional[Path] = None
    xml_path:   Optional[Path] = None
    pdf_path:   Optional[Path] = None
    png_paths:  List[Path]     = field(default_factory=list)
    video_path: Optional[Path] = None
    errors:     List[str]      = field(default_factory=list)
    timings:    dict           = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.midi_path is not None and self.midi_path.exists()


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------

_STEPS = [
    "Decode / normalise audio",
    "Transcribe to MIDI",
    "Generate sheet music",
    "Render tutorial video",
]


class Progress:
    """Thread-safe progress reporter."""

    def __init__(self, cb: Optional[Callable] = None):
        self._cb = cb
        self._step = 0
        self._sub  = 0.0

    def step(self, idx: int, message: str = "") -> None:
        self._step = idx
        self._sub  = 0.0
        pct = idx / len(_STEPS)
        logger.info("[%d/%d] %s", idx, len(_STEPS), message or _STEPS[idx])
        if self._cb:
            self._cb(step=idx, total=len(_STEPS), fraction=pct, message=message)

    def sub(self, fraction: float) -> None:
        self._sub = fraction
        pct = (self._step + fraction) / len(_STEPS)
        if self._cb:
            self._cb(step=self._step, total=len(_STEPS), fraction=pct, message="")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    source,                    # str path/URL | bytes (uploaded file)
    source_name: str,          # original filename (extension used for format detection)
    is_url: bool = False,
    cfg: Optional[PipelineConfig] = None,
    progress_cb: Optional[Callable] = None,
) -> PipelineResult:
    """
    Run the full piano-tutor pipeline.

    Parameters
    ----------
    source      : file path / URL string, or raw bytes from file-uploader
    source_name : display name / filename of the source
    is_url      : True if *source* is a URL
    cfg         : PipelineConfig (defaults used when None)
    progress_cb : optional callback(step, total, fraction, message)

    Returns
    -------
    PipelineResult with populated paths and any error messages
    """
    if cfg is None:
        cfg = PipelineConfig()

    ensure_dir(cfg.cache_root)
    prog  = Progress(progress_cb)
    result = PipelineResult()
    t0_all = time.perf_counter()

    # =========================================================================
    # Step 1 – Decode audio
    # =========================================================================
    prog.step(0, "Decode / normalise audio")
    t0 = time.perf_counter()
    try:
        wav_path, cache_key = prepare_audio(
            source, source_name,
            cache_dir=cfg.cache_root,
            is_url=is_url,
            sample_rate=cfg.sample_rate,
            max_duration=cfg.max_duration,
        )
        result.wav_path = wav_path
        logger.info("WAV ready: %s  (key=%s)", wav_path, cache_key)
    except Exception as e:
        msg = f"Audio decode failed: {e}"
        logger.error(msg)
        result.errors.append(msg)
        return result
    result.timings["decode"] = time.perf_counter() - t0

    out_dir = cache_subdir(cfg.cache_root, cache_key)

    # =========================================================================
    # Step 2 – Transcribe to MIDI
    # =========================================================================
    prog.step(1, "Transcribe to MIDI")
    t0 = time.perf_counter()
    try:
        midi_path, bpm = transcribe(
            wav_path, out_dir,
            mode=cfg.transcription_mode,
            onset_threshold=cfg.onset_threshold,
            frame_threshold=cfg.frame_threshold,
            minimum_note_length=cfg.min_note_ms,
        )
        result.midi_path = midi_path
        logger.info("MIDI: %s  (%.1f BPM)", midi_path, bpm)
        # propagate detected tempo to score config
        cfg.score.tempo_bpm = bpm
    except Exception as e:
        msg = f"Transcription failed: {e}"
        logger.error(msg)
        result.errors.append(msg)
        return result
    result.timings["transcribe"] = time.perf_counter() - t0

    # =========================================================================
    # Step 3 – Sheet music
    # =========================================================================
    prog.step(2, "Generate sheet music")
    t0 = time.perf_counter()
    try:
        cfg.score.quantize = cfg.quantize
        pdf_path, png_paths, xml_path = midi_to_score(midi_path, out_dir / "score", cfg.score)
        result.pdf_path  = pdf_path
        result.png_paths = png_paths
        result.xml_path  = xml_path
        logger.info(
            "Score: pdf=%s  pngs=%d  xml=%s",
            pdf_path, len(png_paths), xml_path
        )
    except Exception as e:
        msg = f"Sheet music generation failed (non-fatal): {e}"
        logger.warning(msg)
        result.errors.append(msg)
        # continue – video can still be rendered
    result.timings["score"] = time.perf_counter() - t0

    # =========================================================================
    # Step 4 – Tutorial video
    # =========================================================================
    prog.step(3, "Render tutorial video")
    t0 = time.perf_counter()
    try:
        video_path = out_dir / "tutorial.mp4"

        def _video_progress(fraction: float) -> None:
            prog.sub(fraction)

        render_falling_notes_video(
            midi_path=midi_path,
            audio_path=wav_path,
            score_png_paths=result.png_paths,
            output_path=video_path,
            cfg=cfg.video,
            progress_cb=_video_progress,
        )
        result.video_path = video_path
    except Exception as e:
        msg = f"Video rendering failed: {e}"
        logger.error(msg, exc_info=True)
        result.errors.append(msg)
    result.timings["video"] = time.perf_counter() - t0
    result.timings["total"] = time.perf_counter() - t0_all

    logger.info(
        "Pipeline done in %.1fs  (decode=%.1f  transcribe=%.1f  score=%.1f  video=%.1f)",
        result.timings["total"],
        result.timings.get("decode", 0),
        result.timings.get("transcribe", 0),
        result.timings.get("score", 0),
        result.timings.get("video", 0),
    )
    return result
