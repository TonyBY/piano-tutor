"""Shared utilities: note names, MIDI helpers, logging, hashing."""
from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# MIDI / music theory constants
# ---------------------------------------------------------------------------

NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAMES_FLAT  = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Semitone indices that are black keys on a piano
BLACK_KEY_CLASSES = frozenset({1, 3, 6, 8, 10})  # C#, D#, F#, G#, A#

PIANO_MIN_MIDI = 21   # A0
PIANO_MAX_MIDI = 108  # C8


def midi_to_note_name(midi: int, use_flats: bool = False) -> str:
    """Return note name like 'C4', 'F#3', etc."""
    names = NOTE_NAMES_FLAT if use_flats else NOTE_NAMES_SHARP
    octave = (midi // 12) - 1
    return f"{names[midi % 12]}{octave}"


def note_name_to_midi(name: str) -> int:
    """Convert 'C4' → 60, 'F#3' → 54, etc."""
    import re
    m = re.match(r"([A-G][b#]?)(-?\d+)", name.strip())
    if not m:
        raise ValueError(f"Cannot parse note name: {name!r}")
    note_str, octave_str = m.group(1), m.group(2)
    try:
        idx = NOTE_NAMES_SHARP.index(note_str)
    except ValueError:
        idx = NOTE_NAMES_FLAT.index(note_str)
    return (int(octave_str) + 1) * 12 + idx


def is_black_key(midi: int) -> bool:
    return (midi % 12) in BLACK_KEY_CLASSES


# ---------------------------------------------------------------------------
# File hashing (for cache keys)
# ---------------------------------------------------------------------------

def compute_file_hash(path: Path, length: int = 16) -> str:
    """Return first *length* hex chars of SHA-256 of file content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def compute_bytes_hash(data: bytes, length: int = 16) -> str:
    return hashlib.sha256(data).hexdigest()[:length]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_subdir(cache_root: Path, key: str) -> Path:
    d = cache_root / key
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# System-tool discovery
# ---------------------------------------------------------------------------

import shutil
import os
import subprocess


def find_executable(*candidates: str) -> Optional[str]:
    """Return the first executable found in PATH or common install dirs."""
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
    return None


def find_musescore() -> Optional[str]:
    found = find_executable("mscore4", "mscore3", "mscore", "musescore4", "musescore3", "musescore")
    if found:
        return found
    # macOS app bundles
    mac_candidates = [
        "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
        "/Applications/MuseScore 3.app/Contents/MacOS/mscore3",
        "/Applications/MuseScore4.app/Contents/MacOS/mscore",
    ]
    for p in mac_candidates:
        if os.path.isfile(p):
            return p
    return None


def find_lilypond() -> Optional[str]:
    found = find_executable("lilypond")
    if found:
        return found
    mac_candidates = [
        "/Applications/LilyPond.app/Contents/Resources/bin/lilypond",
    ]
    for p in mac_candidates:
        if os.path.isfile(p):
            return p
    return None


def find_ffmpeg() -> str:
    p = find_executable("ffmpeg")
    if p:
        return p
    raise EnvironmentError(
        "ffmpeg not found. Install with: brew install ffmpeg  OR  apt install ffmpeg"
    )


def check_tool_version(cmd: list[str]) -> str:
    """Run cmd and return stdout+stderr (first line). Silent on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return (r.stdout + r.stderr).strip().splitlines()[0]
    except Exception:
        return "unavailable"


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))
