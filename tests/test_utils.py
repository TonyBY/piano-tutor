"""Tests for src/utils.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.utils import (
    midi_to_note_name,
    note_name_to_midi,
    is_black_key,
    compute_bytes_hash,
    BLACK_KEY_CLASSES,
)


class TestMidiNoteNames:
    def test_middle_c(self):
        assert midi_to_note_name(60) == "C4"

    def test_a4(self):
        assert midi_to_note_name(69) == "A4"

    def test_a0_lowest(self):
        assert midi_to_note_name(21) == "A0"

    def test_c8_highest(self):
        assert midi_to_note_name(108) == "C8"

    def test_f_sharp(self):
        assert midi_to_note_name(66) == "F#4"

    def test_flat_mode(self):
        assert midi_to_note_name(66, use_flats=True) == "Gb4"


class TestNoteNameToMidi:
    def test_c4(self):
        assert note_name_to_midi("C4") == 60

    def test_a0(self):
        assert note_name_to_midi("A0") == 21

    def test_f_sharp_3(self):
        assert note_name_to_midi("F#3") == 54

    def test_round_trip(self):
        for midi in [21, 36, 48, 60, 72, 84, 108]:
            assert note_name_to_midi(midi_to_note_name(midi)) == midi


class TestBlackKey:
    def test_c_is_white(self):
        assert not is_black_key(60)

    def test_c_sharp_is_black(self):
        assert is_black_key(61)

    def test_d_sharp_is_black(self):
        assert is_black_key(63)

    def test_e_is_white(self):
        assert not is_black_key(64)

    def test_black_key_classes(self):
        assert BLACK_KEY_CLASSES == {1, 3, 6, 8, 10}


class TestHashing:
    def test_deterministic(self):
        data = b"hello world"
        h1 = compute_bytes_hash(data)
        h2 = compute_bytes_hash(data)
        assert h1 == h2

    def test_different_data_different_hash(self):
        h1 = compute_bytes_hash(b"hello")
        h2 = compute_bytes_hash(b"world")
        assert h1 != h2

    def test_length(self):
        h = compute_bytes_hash(b"test", length=8)
        assert len(h) == 8
