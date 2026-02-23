"""Tests for src/render_video.py – keyboard layout and frame rendering."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

# Skip entire module if heavy deps are absent
cv2 = pytest.importorskip("cv2", reason="opencv-python-headless not installed")
np = pytest.importorskip("numpy", reason="numpy not installed")

from src.render_video import (
    VideoConfig,
    build_key_layout,
    render_keyboard_image,
    highlight_active_keys,
    prepare_score_strip,
    _note_color,
)
from src.utils import BLACK_KEY_CLASSES, PIANO_MIN_MIDI, PIANO_MAX_MIDI


class TestKeyLayout:
    """Test the piano key layout builder."""

    def setup_method(self):
        self.cfg = VideoConfig(video_width=1280, video_height=720, fps=30)
        self.keys = build_key_layout(1280, (PIANO_MIN_MIDI, PIANO_MAX_MIDI))

    def test_correct_total_keys(self):
        # Full piano: A0 (21) to C8 (108) = 88 keys
        assert len(self.keys) == 88

    def test_correct_note_range(self):
        notes = [k["note"] for k in self.keys]
        assert min(notes) == PIANO_MIN_MIDI
        assert max(notes) == PIANO_MAX_MIDI

    def test_white_key_count(self):
        whites = [k for k in self.keys if not k["is_black"]]
        assert len(whites) == 52

    def test_black_key_count(self):
        blacks = [k for k in self.keys if k["is_black"]]
        assert len(blacks) == 36

    def test_black_key_narrower_than_white(self):
        white_w = next(k["w"] for k in self.keys if not k["is_black"])
        for k in self.keys:
            if k["is_black"]:
                assert k["w"] < white_w

    def test_x_positions_monotone(self):
        """White keys should be left-to-right in order."""
        white_xs = [k["x"] for k in self.keys if not k["is_black"]]
        for a, b in zip(white_xs, white_xs[1:]):
            assert b > a, "White keys not monotonically increasing"

    def test_all_keys_in_bounds(self):
        for k in self.keys:
            assert k["x"] >= 0
            assert k["x"] + k["w"] <= 1280 + 5   # tiny float rounding OK

    def test_color_is_bgr_tuple(self):
        for k in self.keys:
            c = k["color"]
            assert len(c) == 3
            assert all(0 <= v <= 255 for v in c)

    def test_pitch_class_black_keys(self):
        for k in self.keys:
            if k["is_black"]:
                assert k["note"] % 12 in BLACK_KEY_CLASSES
            else:
                assert k["note"] % 12 not in BLACK_KEY_CLASSES

    def test_small_range(self):
        """Single octave C4–C5."""
        keys = build_key_layout(700, (60, 72))
        notes = [k["note"] for k in keys]
        assert 60 in notes and 72 in notes
        assert len(keys) == 13


class TestKeyboardImage:
    def test_shape(self):
        cfg = VideoConfig(video_width=1280, video_height=720, keyboard_height=140)
        keys = build_key_layout(1280, (PIANO_MIN_MIDI, PIANO_MAX_MIDI))
        img = render_keyboard_image(keys, cfg)
        assert img.shape == (140, 1280, 3)
        assert img.dtype == np.uint8

    def test_has_content(self):
        cfg = VideoConfig(video_width=640, video_height=480, keyboard_height=120)
        keys = build_key_layout(640, (48, 84))
        img = render_keyboard_image(keys, cfg)
        # Should not be all zeros
        assert img.max() > 0

    def test_highlight_changes_image(self):
        cfg = VideoConfig(video_width=640, video_height=480, keyboard_height=120)
        keys = build_key_layout(640, (48, 84))
        base = render_keyboard_image(keys, cfg)
        highlighted = highlight_active_keys(base, keys, {60}, cfg)
        # Images should differ
        assert not np.array_equal(base, highlighted)

    def test_highlight_no_active(self):
        cfg = VideoConfig(video_width=640, video_height=480, keyboard_height=120)
        keys = build_key_layout(640, (48, 84))
        base = render_keyboard_image(keys, cfg)
        highlighted = highlight_active_keys(base, keys, set(), cfg)
        assert np.array_equal(base, highlighted)


class TestNoteColor:
    def test_returns_bgr_triple(self):
        for midi in [21, 60, 72, 108]:
            c = _note_color(midi)
            assert len(c) == 3
            assert all(0 <= v <= 255 for v in c)


class TestScoreStrip:
    def test_returns_none_on_empty_list(self):
        result = prepare_score_strip([], strip_height=180, video_width=1280)
        assert result is None

    def test_returns_none_on_missing_files(self, tmp_path):
        result = prepare_score_strip(
            [tmp_path / "nonexistent.png"], strip_height=180, video_width=1280
        )
        assert result is None

    def test_valid_strip(self, tmp_path):
        import cv2

        # Create a dummy PNG
        dummy = np.zeros((600, 800, 3), dtype=np.uint8)
        dummy[:, :] = (200, 200, 200)
        png_path = tmp_path / "page1.png"
        cv2.imwrite(str(png_path), dummy)

        strip = prepare_score_strip([png_path], strip_height=180, video_width=1280)
        assert strip is not None
        assert strip.shape[0] == 180
        assert strip.shape[2] == 3
