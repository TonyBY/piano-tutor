"""Audio → MIDI transcription.

Primary path : Spotify Basic Pitch (polyphonic, open-source, CPU-friendly)
Fallback path: librosa pYIN (monophonic)
"""
from __future__ import annotations

import bisect
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from .utils import ensure_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Basic Pitch (polyphonic)
# ---------------------------------------------------------------------------

def _try_basic_pitch(
    wav_path: Path,
    output_dir: Path,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 58.0,   # ms
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = None,
) -> Path:
    """
    Transcribe with Basic Pitch.  Returns path to MIDI file.
    Supports basic-pitch >= 0.3 (ONNX) and 0.2.x (TF).
    """
    ensure_dir(output_dir)
    out_midi = output_dir / f"{wav_path.stem}_basic_pitch.mid"

    if out_midi.exists() and out_midi.stat().st_size > 0:
        logger.info("Using cached Basic Pitch MIDI: %s", out_midi)
        return out_midi

    logger.info("Running Basic Pitch on %s …", wav_path.name)

    try:
        # ── Try modern API (0.3+, ONNX) ──────────────────────────────────
        from basic_pitch.inference import predict        # noqa: PLC0415
        from basic_pitch import ICASSP_2022_MODEL_PATH  # noqa: PLC0415

        kw = dict(
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
            melodia_trick=True,
            multiple_pitch_bends=False,
        )
        if minimum_frequency is not None:
            kw["minimum_frequency"] = minimum_frequency
        if maximum_frequency is not None:
            kw["maximum_frequency"] = maximum_frequency

        model_output, midi_data, _ = predict(str(wav_path), ICASSP_2022_MODEL_PATH, **kw)
        midi_data.write(str(out_midi))
        logger.info("Basic Pitch: wrote %s", out_midi)
        return out_midi

    except TypeError:
        # Older API: predict_and_save
        logger.debug("predict() signature mismatch – trying predict_and_save()")
        from basic_pitch.inference import predict_and_save  # noqa: PLC0415
        from basic_pitch import ICASSP_2022_MODEL_PATH      # noqa: PLC0415

        predict_and_save(
            [str(wav_path)],
            str(output_dir),
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            melodia_trick=True,
            multiple_pitch_bends=False,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
        )
        # find the output file
        candidates = sorted(output_dir.glob("*_basic_pitch.mid"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]
        raise RuntimeError("predict_and_save ran but no MIDI file was produced.")


# ---------------------------------------------------------------------------
# pYIN fallback (monophonic)
# ---------------------------------------------------------------------------

def _transcribe_pyin(wav_path: Path, output_dir: Path) -> Path:
    """
    Monophonic pitch tracker via librosa.pyin.
    Produces a single-track MIDI file.
    """
    import librosa          # noqa: PLC0415
    import pretty_midi      # noqa: PLC0415

    ensure_dir(output_dir)
    out_midi = output_dir / f"{wav_path.stem}_pyin.mid"
    if out_midi.exists() and out_midi.stat().st_size > 0:
        logger.info("Using cached pYIN MIDI: %s", out_midi)
        return out_midi

    logger.info("Running pYIN on %s …", wav_path.name)
    y, sr = librosa.load(str(wav_path), sr=None)

    hop = 512
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sr,
        frame_length=2048,
        hop_length=hop,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop)

    # Convert f0 sequence → pretty_midi notes
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    piano = pretty_midi.Instrument(program=0, name="Piano")

    in_note = False
    note_start = 0.0
    cur_pitch = 0
    MIN_DUR = 0.05  # seconds; discard shorter notes

    for t, freq, voiced in zip(times, f0, voiced_flag):
        if voiced and freq is not None and not np.isnan(freq) and freq > 0:
            pitch = int(np.clip(np.round(librosa.hz_to_midi(freq)), 21, 108))
            if not in_note:
                in_note = True
                note_start = float(t)
                cur_pitch = pitch
            elif pitch != cur_pitch:
                dur = float(t) - note_start
                if dur >= MIN_DUR:
                    piano.notes.append(pretty_midi.Note(80, cur_pitch, note_start, float(t)))
                note_start = float(t)
                cur_pitch = pitch
        else:
            if in_note:
                dur = float(t) - note_start
                if dur >= MIN_DUR:
                    piano.notes.append(pretty_midi.Note(80, cur_pitch, note_start, float(t)))
                in_note = False

    if in_note and len(times) > 0:
        dur = float(times[-1]) - note_start
        if dur >= MIN_DUR:
            piano.notes.append(pretty_midi.Note(80, cur_pitch, note_start, float(times[-1])))

    pm.instruments.append(piano)
    pm.write(str(out_midi))
    logger.info("pYIN: wrote %s  (%d notes)", out_midi, len(piano.notes))
    return out_midi


# ---------------------------------------------------------------------------
# Tempo detection helpers
# ---------------------------------------------------------------------------

def _tempo_from_midi(midi_path: Path) -> Optional[float]:
    """Read the first tempo marking from a MIDI file (fast, no audio load)."""
    try:
        import pretty_midi  # noqa: PLC0415
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        tempos = pm.get_tempo_changes()  # returns (times, tempos)
        if tempos[1].size > 0:
            bpm = float(tempos[1][0])
            bpm = max(40.0, min(220.0, bpm))
            logger.info("Tempo from MIDI: %.1f BPM", bpm)
            return bpm
    except Exception as e:
        logger.debug("MIDI tempo read failed: %s", e)
    return None


def detect_tempo(wav_path: Path) -> float:
    """Estimate BPM via librosa beat tracker (loads audio — only used as fallback)."""
    try:
        import librosa  # noqa: PLC0415
        # Load only the first 30s — sufficient for beat tracking, avoids slow full load
        y, sr = librosa.load(str(wav_path), sr=None, duration=30.0)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        bpm = max(40.0, min(220.0, bpm))
        logger.info("Detected tempo (librosa): %.1f BPM", bpm)
        return bpm
    except Exception as e:
        logger.warning("Tempo detection failed (%s); using 120 BPM", e)
        return 120.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe(
    wav_path: Path,
    output_dir: Path,
    mode: str = "polyphonic",
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 58.0,
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = None,
) -> tuple[Path, float]:
    """
    Transcribe *wav_path* to MIDI.

    Parameters
    ----------
    wav_path          : path to mono WAV file
    output_dir        : where to write .mid file
    mode              : 'polyphonic' (Basic Pitch) | 'monophonic' (pYIN)
    onset_threshold   : Basic Pitch onset confidence threshold
    frame_threshold   : Basic Pitch frame confidence threshold
    minimum_note_length: minimum note length in ms (Basic Pitch)

    Returns
    -------
    (midi_path, detected_bpm)
    """
    if mode == "polyphonic":
        try:
            midi_path = _try_basic_pitch(
                wav_path, output_dir,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                minimum_note_length=minimum_note_length,
                minimum_frequency=minimum_frequency,
                maximum_frequency=maximum_frequency,
            )
            # Get tempo from the MIDI output — avoids a redundant librosa.load
            bpm = _tempo_from_midi(midi_path) or detect_tempo(wav_path)
            return midi_path, bpm
        except Exception as e:
            logger.warning(
                "Basic Pitch failed (%s) – falling back to pYIN (monophonic).", e
            )

    # pYIN already loads audio; detect_tempo is only called here for monophonic path
    midi_path = _transcribe_pyin(wav_path, output_dir)
    bpm = _tempo_from_midi(midi_path) or detect_tempo(wav_path)
    return midi_path, bpm
