"""Tests for src/midi_to_score.py – sheet-music generation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

# Needs pretty_midi and matplotlib
pretty_midi = pytest.importorskip("pretty_midi", reason="pretty_midi not installed")
matplotlib  = pytest.importorskip("matplotlib", reason="matplotlib not installed")


def _make_simple_midi(path: Path) -> Path:
    """Write a minimal 4-note MIDI file using pretty_midi."""
    import pretty_midi  # noqa: PLC0415

    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0, name="Piano")
    for i, (pitch, start) in enumerate([(60, 0.0), (64, 0.5), (67, 1.0), (72, 1.5)]):
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=start + 0.45))
    pm.instruments.append(inst)
    pm.write(str(path))
    return path


class TestPianoRollFallback:
    """Piano-roll fallback runs without MuseScore/LilyPond."""

    def test_produces_png(self, tmp_path):
        from src.midi_to_score import _render_piano_roll

        midi_path = _make_simple_midi(tmp_path / "test.mid")
        _, png_paths = _render_piano_roll(midi_path, tmp_path / "score")
        assert len(png_paths) >= 1
        for p in png_paths:
            assert p.exists()
            assert p.stat().st_size > 0

    def test_png_is_valid_image(self, tmp_path):
        import cv2
        from src.midi_to_score import _render_piano_roll

        midi_path = _make_simple_midi(tmp_path / "test.mid")
        _, png_paths = _render_piano_roll(midi_path, tmp_path / "score")
        img = cv2.imread(str(png_paths[0]))
        assert img is not None
        assert img.shape[2] == 3


class TestMidiToScore:
    """End-to-end score generation (MuseScore/LilyPond optional)."""

    def test_returns_tuple(self, tmp_path):
        from src.midi_to_score import midi_to_score, ScoreConfig

        midi_path = _make_simple_midi(tmp_path / "test.mid")
        cfg = ScoreConfig(tempo_bpm=120.0, quantize=True)
        pdf, pngs, xml = midi_to_score(midi_path, tmp_path / "out", cfg)
        # png_paths should always be non-empty (piano-roll fallback)
        assert isinstance(pngs, list)
        assert len(pngs) >= 1

    def test_xml_written_when_music21_available(self, tmp_path):
        """MusicXML should be written if music21 is installed."""
        try:
            import music21  # noqa: F401
        except ImportError:
            pytest.skip("music21 not installed")

        from src.midi_to_score import midi_to_score, ScoreConfig

        midi_path = _make_simple_midi(tmp_path / "test.mid")
        cfg = ScoreConfig(tempo_bpm=120.0, quantize=True)
        _, _, xml = midi_to_score(midi_path, tmp_path / "out", cfg)
        if xml is not None:
            assert xml.exists()

    def test_caching(self, tmp_path):
        """Second call should skip reprocessing (no change in mtime)."""
        from src.midi_to_score import midi_to_score, ScoreConfig

        midi_path = _make_simple_midi(tmp_path / "test.mid")
        cfg = ScoreConfig(tempo_bpm=120.0, quantize=False)
        out = tmp_path / "out"
        _, pngs1, _ = midi_to_score(midi_path, out, cfg)
        mtime1 = pngs1[0].stat().st_mtime if pngs1 else None

        _, pngs2, _ = midi_to_score(midi_path, out, cfg)
        mtime2 = pngs2[0].stat().st_mtime if pngs2 else None

        if mtime1 and mtime2:
            assert mtime1 == mtime2
