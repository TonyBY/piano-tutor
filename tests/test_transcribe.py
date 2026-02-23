"""Tests for src/transcribe.py – MIDI transcription."""
import sys
import struct
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
np = pytest.importorskip("numpy", reason="numpy not installed")
librosa = pytest.importorskip("librosa", reason="librosa not installed")


def _make_sine_wav(path: Path, freq_hz: float = 440.0, duration: float = 2.0,
                   sr: int = 22050) -> Path:
    """Generate a pure-tone WAV (mono, 16-bit)."""
    import numpy as _np
    t = _np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (_np.sin(2 * _np.pi * freq_hz * t) * 32767 * 0.8).astype(_np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(signal.tobytes())
    return path


class TestPyinFallback:
    """pYIN transcription should always run (no external deps)."""

    def test_single_tone_produces_midi(self, tmp_path):
        from src.transcribe import _transcribe_pyin

        wav = _make_sine_wav(tmp_path / "a4.wav", freq_hz=440.0, duration=2.0)
        midi_path = _transcribe_pyin(wav, tmp_path)

        assert midi_path.exists()
        assert midi_path.stat().st_size > 0

    def test_single_tone_note_count(self, tmp_path):
        """A steady 440 Hz tone should yield ≥ 1 note near A4 (MIDI 69)."""
        import pretty_midi
        from src.transcribe import _transcribe_pyin

        wav = _make_sine_wav(tmp_path / "a4.wav", freq_hz=440.0, duration=2.0)
        midi_path = _transcribe_pyin(wav, tmp_path)
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        notes = [n for inst in pm.instruments for n in inst.notes]
        assert len(notes) >= 1
        # All notes should be near A4 (MIDI 69 ± 3 semitones)
        for n in notes:
            assert abs(n.pitch - 69) <= 3, f"Unexpected pitch {n.pitch}"

    def test_caching(self, tmp_path):
        """Second call returns cached file without re-processing."""
        from src.transcribe import _transcribe_pyin

        wav = _make_sine_wav(tmp_path / "test.wav", freq_hz=330.0, duration=1.5)
        p1 = _transcribe_pyin(wav, tmp_path)
        mtime1 = p1.stat().st_mtime
        p2 = _transcribe_pyin(wav, tmp_path)
        mtime2 = p2.stat().st_mtime
        assert mtime1 == mtime2


class TestTempoDetection:
    def test_returns_float(self, tmp_path):
        from src.transcribe import detect_tempo

        wav = _make_sine_wav(tmp_path / "test.wav", freq_hz=440.0, duration=3.0)
        bpm = detect_tempo(wav)
        assert isinstance(bpm, float)
        assert 40.0 <= bpm <= 220.0

    def test_nonexistent_file_returns_default(self, tmp_path):
        """Should not crash; returns 120.0 on error."""
        from src.transcribe import detect_tempo

        bpm = detect_tempo(tmp_path / "no_such_file.wav")
        assert bpm == 120.0
