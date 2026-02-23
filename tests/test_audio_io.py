"""Tests for src/audio_io.py – audio decode helpers."""
import sys
import struct
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.audio_io import decode_to_wav, prepare_audio, _looks_like_direct_audio_url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_silent_wav(path: Path, duration_s: float = 1.0, sr: int = 22050) -> Path:
    """Write a minimal silent WAV to *path*."""
    n_samples = int(sr * duration_s)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * n_samples)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUrlDetection:
    def test_mp3_url(self):
        assert _looks_like_direct_audio_url("https://example.com/song.mp3")

    def test_wav_url(self):
        assert _looks_like_direct_audio_url("https://example.com/audio.wav?token=abc")

    def test_youtube_url(self):
        assert not _looks_like_direct_audio_url("https://www.youtube.com/watch?v=abc")

    def test_html_url(self):
        assert not _looks_like_direct_audio_url("https://example.com/page.html")


class TestDecodeToWav:
    def test_wav_passthrough(self, tmp_path):
        """A WAV file should decode without error (may re-encode to normalise)."""
        src = _make_silent_wav(tmp_path / "input.wav")
        out = decode_to_wav(src, cache_dir=tmp_path / "cache", sample_rate=22050)
        assert out.exists()
        assert out.stat().st_size > 44  # more than just WAV header

    def test_caching(self, tmp_path):
        """Calling decode_to_wav twice returns the same cached file."""
        src = _make_silent_wav(tmp_path / "input.wav")
        cache = tmp_path / "cache"
        out1 = decode_to_wav(src, cache_dir=cache, sample_rate=22050)
        mtime1 = out1.stat().st_mtime
        out2 = decode_to_wav(src, cache_dir=cache, sample_rate=22050)
        mtime2 = out2.stat().st_mtime
        assert out1 == out2
        assert mtime1 == mtime2  # not re-processed


class TestPrepareAudio:
    def test_bytes_source(self, tmp_path):
        """Bytes (simulating file upload) should be saved and decoded."""
        wav_file = _make_silent_wav(tmp_path / "test.wav")
        audio_bytes = wav_file.read_bytes()
        wav_path, key = prepare_audio(
            audio_bytes,
            source_name="test.wav",
            cache_dir=tmp_path / "cache",
        )
        assert wav_path.exists()
        assert isinstance(key, str) and len(key) == 16

    def test_file_path_source(self, tmp_path):
        """File path source should be decoded."""
        wav_file = _make_silent_wav(tmp_path / "test.wav")
        wav_path, key = prepare_audio(
            wav_file,
            source_name="test.wav",
            cache_dir=tmp_path / "cache",
        )
        assert wav_path.exists()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            prepare_audio(
                tmp_path / "nonexistent.wav",
                source_name="nonexistent.wav",
                cache_dir=tmp_path / "cache",
            )
