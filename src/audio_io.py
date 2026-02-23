"""Audio I/O: download from URL, decode to normalised WAV via ffmpeg.

YouTube 403 strategy
────────────────────
YouTube increasingly blocks yt-dlp without browser cookies.
We try four escalating strategies, stopping at the first success:

  1. android_vr player client  – bypasses most 403s without cookies
  2. tv_embedded player client – alternative bypass
  3. Cookies from Chrome/Firefox (if a browser is installed on the machine)
  4. Direct urllib download    – for plain https:// audio file URLs only

If all fail, we raise a descriptive error with troubleshooting hints.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from .utils import compute_file_hash, compute_bytes_hash, ensure_dir, find_ffmpeg

logger = logging.getLogger(__name__)

# Supported upload extensions that ffmpeg can decode
SUPPORTED_EXTENSIONS = {
    ".mp3", ".mp4", ".m4a", ".aac", ".ogg", ".opus",
    ".flac", ".wav", ".wma", ".webm", ".aif", ".aiff",
}

# yt-dlp player-client strategies to try in order (most reliable first)
_YTDLP_STRATEGIES: List[Tuple[str, List[str]]] = [
    (
        "android_vr",
        ["--extractor-args", "youtube:player_client=android_vr"],
    ),
    (
        "tv_embedded",
        ["--extractor-args", "youtube:player_client=tv_embedded"],
    ),
    (
        "mweb",
        ["--extractor-args", "youtube:player_client=mweb"],
    ),
]

# Browsers to try for cookie extraction (yt-dlp --cookies-from-browser)
_COOKIE_BROWSERS = ["chrome", "firefox", "edge", "safari", "brave", "chromium"]


# ---------------------------------------------------------------------------
# URL download
# ---------------------------------------------------------------------------

def download_from_url(url: str, cache_dir: Path) -> Path:
    """
    Download audio from a URL.  Returns path to the downloaded file.

    Tries yt-dlp with multiple player-client strategies, then browser cookies,
    then a plain urllib download for direct audio file URLs.
    """
    ensure_dir(cache_dir)
    url_hash = compute_bytes_hash(url.encode())
    output_template = str(cache_dir / f"{url_hash}.%(ext)s")

    # ── Cache hit ─────────────────────────────────────────────────────────
    existing = [p for p in cache_dir.glob(f"{url_hash}.*") if p.is_file()]
    if existing:
        logger.info("URL already cached: %s", existing[0])
        return existing[0]

    logger.info("Downloading: %s", url)

    # ── Base yt-dlp flags (shared by all attempts) ────────────────────────
    base_flags = [
        "yt-dlp",
        "--no-playlist",
        "-x",                        # extract audio only
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--no-warnings",
        "--no-check-certificates",   # skip SSL verification issues
        "-o", output_template,
    ]

    errors: List[str] = []

    # ── Strategy 1-3: player-client overrides ─────────────────────────────
    for strategy_name, extra_flags in _YTDLP_STRATEGIES:
        cmd = base_flags + extra_flags + [url]
        logger.debug("yt-dlp attempt [%s]: %s", strategy_name, " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            found = _find_downloaded_file(result.stdout, cache_dir, url_hash)
            if found:
                logger.info("Downloaded via strategy '%s': %s", strategy_name, found)
                return found

        err = (result.stderr or result.stdout)[-600:].strip()
        # Suppress the Python 3.8 deprecation line from error reporting
        err = "\n".join(
            l for l in err.splitlines()
            if "Deprecated Feature: Support for Python" not in l
        )
        errors.append(f"[{strategy_name}] {err}")
        logger.debug("Strategy '%s' failed: %s", strategy_name, err[-200:])

    # ── Strategy 4: browser cookies ───────────────────────────────────────
    for browser in _COOKIE_BROWSERS:
        if shutil.which(browser) is None and browser not in ("chrome", "firefox",
                                                              "safari", "edge"):
            continue  # Skip browsers not obviously installed
        cmd = base_flags + ["--cookies-from-browser", browser, url]
        logger.debug("yt-dlp attempt [cookies:%s]", browser)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            found = _find_downloaded_file(result.stdout, cache_dir, url_hash)
            if found:
                logger.info("Downloaded via browser cookies (%s): %s", browser, found)
                return found

        err = (result.stderr or "")[-200:].strip()
        errors.append(f"[cookies:{browser}] {err}")

    # ── Strategy 5: plain urllib (direct audio URLs only) ─────────────────
    if _looks_like_direct_audio_url(url):
        try:
            return _direct_download(url, cache_dir, url_hash)
        except Exception as e:
            errors.append(f"[direct-download] {e}")

    # ── All strategies failed ─────────────────────────────────────────────
    raise _make_403_error(url, errors)


def _find_downloaded_file(stdout: str, cache_dir: Path, url_hash: str) -> Optional[Path]:
    """Locate the file yt-dlp wrote, from its stdout or by globbing."""
    for line in stdout.splitlines():
        for marker in ("[ExtractAudio] Destination:", "[download] Destination:",
                       "Destination:", "Merging formats into"):
            if marker in line:
                p = Path(line.split(marker)[-1].strip().strip('"'))
                if p.exists() and p.stat().st_size > 0:
                    return p
    # Glob fallback
    candidates = [
        p for p in sorted(cache_dir.glob(f"{url_hash}.*"),
                          key=lambda x: x.stat().st_mtime)
        if p.is_file() and p.stat().st_size > 0
    ]
    return candidates[-1] if candidates else None


def _looks_like_direct_audio_url(url: str) -> bool:
    ext = Path(url.split("?")[0]).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS


def _direct_download(url: str, cache_dir: Path, stem: str) -> Path:
    """Fallback: download a direct-link audio file via urllib."""
    import urllib.request  # noqa: PLC0415
    import urllib.error    # noqa: PLC0415

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    req = urllib.request.Request(url, headers=headers)
    logger.info("Direct download: %s", url)
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()

    tmp = cache_dir / f"{stem}_raw"
    tmp.write_bytes(data)
    out = cache_dir / f"{stem}.wav"
    _decode_to_wav(tmp, out, sample_rate=22050, mono=True)
    tmp.unlink(missing_ok=True)
    return out


def _make_403_error(url: str, errors: List[str]) -> RuntimeError:
    is_youtube = any(h in url for h in ("youtube.com", "youtu.be"))
    hint = ""
    if is_youtube:
        hint = (
            "\n\nYouTube 403 troubleshooting:\n"
            "  1. Open Chrome/Firefox and log into YouTube (no incognito).\n"
            "  2. The app will automatically try to use your browser cookies.\n"
            "  3. Or download the audio manually and use the 'Upload file' tab.\n"
            "  4. Or use a SoundCloud / direct MP3 URL instead."
        )
    summary = "\n".join(f"  • {e}" for e in errors[:6])
    return RuntimeError(
        f"All download strategies failed for:\n  {url}\n\n"
        f"Errors:\n{summary}"
        f"{hint}"
    )


# ---------------------------------------------------------------------------
# Decode / normalise
# ---------------------------------------------------------------------------

def decode_to_wav(
    input_path: Path,
    cache_dir: Path,
    sample_rate: int = 22050,
    mono: bool = True,
    max_duration: Optional[float] = None,
) -> Path:
    """Decode *input_path* to 16-bit WAV at *sample_rate* Hz, with caching.

    If *max_duration* is set, only the first *max_duration* seconds are kept.
    This is the single most effective way to speed up transcription on long files.
    """
    ensure_dir(cache_dir)
    file_hash = compute_file_hash(input_path)
    dur_tag = f"_t{int(max_duration)}" if max_duration else ""
    out = cache_dir / f"{file_hash}_sr{sample_rate}{'_mono' if mono else ''}{dur_tag}.wav"

    if out.exists() and out.stat().st_size > 0:
        logger.info("Using cached WAV: %s", out)
        return out

    _decode_to_wav(input_path, out, sample_rate=sample_rate, mono=mono,
                   max_duration=max_duration)
    return out


def _decode_to_wav(
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    mono: bool,
    max_duration: Optional[float] = None,
) -> None:
    ffmpeg = find_ffmpeg()
    cmd = [ffmpeg, "-y", "-i", str(input_path)]
    if max_duration is not None and max_duration > 0:
        cmd += ["-t", str(max_duration)]
    if mono:
        cmd += ["-ac", "1"]
    cmd += [
        "-ar", str(sample_rate),
        "-sample_fmt", "s16",
        "-f", "wav",
        str(output_path),
    ]
    logger.info("Decoding: %s → %s%s", input_path.name, output_path.name,
                f" (capped at {max_duration:.0f}s)" if max_duration else "")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed:\n{result.stderr[-2000:]}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg produced an empty output file.")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def prepare_audio(
    source,
    source_name: str,
    cache_dir: Path,
    is_url: bool = False,
    sample_rate: int = 22050,
    max_duration: Optional[float] = None,
) -> Tuple[Path, str]:
    """
    Obtain audio from *source* and return (wav_path, cache_key).

    *source* can be:
      - str / Path : local file path or URL
      - bytes       : raw bytes from st.file_uploader

    *max_duration*: if set, only the first N seconds are decoded (fastest trim).
    """
    ensure_dir(cache_dir)

    if is_url:
        raw_path = download_from_url(str(source), cache_dir)
    elif isinstance(source, (bytes, bytearray)):
        data_hash = compute_bytes_hash(bytes(source))
        ext = Path(source_name).suffix.lower() or ".audio"
        raw_path = cache_dir / f"{data_hash}{ext}"
        if not raw_path.exists():
            raw_path.write_bytes(bytes(source))
        logger.info("Saved upload to %s", raw_path)
    else:
        raw_path = Path(source)

    if not raw_path.exists():
        raise FileNotFoundError(f"Audio source not found: {raw_path}")

    wav_path = decode_to_wav(raw_path, cache_dir, sample_rate=sample_rate,
                             max_duration=max_duration)
    cache_key = compute_file_hash(wav_path)
    return wav_path, cache_key


# ---------------------------------------------------------------------------
# Duration probe
# ---------------------------------------------------------------------------

def probe_duration(wav_path: Path) -> float:
    """Return duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(wav_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return float(r.stdout.strip())
    except Exception:
        return 0.0
