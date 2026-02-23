"""Piano Tutor – Streamlit web application.

Run locally:
    streamlit run app.py

Upload an audio clip (or paste a URL) and get back:
  • MIDI file
  • Sheet-music PDF + PNG preview
  • Falling-notes tutorial video (MP4)
"""
from __future__ import annotations

import io
import logging
import sys
from typing import Optional
import time
from pathlib import Path

import streamlit as st

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging, find_musescore, find_lilypond, find_ffmpeg
from src.pipeline import PipelineConfig, run_pipeline, _STEPS
from src.render_video import VideoConfig
from src.midi_to_score import ScoreConfig

setup_logging(logging.WARNING)  # suppress verbose library logs in Streamlit

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎹 Piano Tutor",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – advanced settings
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("Transcription")
    max_dur_s = st.slider(
        "Max audio duration (s)", 15, 300, 120, 15,
        help=(
            "Clip audio to this length before processing. "
            "Lower = much faster. 120 s is a good default for a full piece excerpt."
        ),
    )
    trans_mode = st.selectbox(
        "Mode",
        ["polyphonic (Basic Pitch)", "monophonic (pYIN fallback)"],
        index=0,
        help=(
            "Polyphonic uses Spotify Basic Pitch – works for chords.\n"
            "Monophonic uses librosa pYIN – faster, single note at a time."
        ),
    )
    onset_thr = st.slider("Onset threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Lower = more notes detected (may add noise)")
    frame_thr = st.slider("Frame threshold", 0.1, 0.9, 0.3, 0.05)
    min_note_ms = st.slider("Min note length (ms)", 20, 300, 58, 5)
    quantize   = st.checkbox("Quantise score", value=True,
                             help="Snap notes to nearest 8th-note grid for cleaner notation")

    st.subheader("Video")
    res_choice = st.selectbox("Resolution", ["1280 × 720", "1920 × 1080", "854 × 480"])
    fps        = st.slider("FPS", 15, 60, 30, 5)
    fall_speed = st.slider("Fall speed (px/s)", 50, 400, 200, 25)
    key_lo     = st.number_input("Lowest MIDI note (A0 = 21)",  min_value=21,  max_value=60,  value=21)
    key_hi     = st.number_input("Highest MIDI note (C8 = 108)", min_value=60, max_value=108, value=108)

    st.subheader("Environment")
    mscore_path  = find_musescore()
    lilypond_path= find_lilypond()
    try:
        ffmpeg_path  = find_ffmpeg()
        st.success(f"✅ ffmpeg found")
    except EnvironmentError:
        st.error("❌ ffmpeg NOT found – please install it")
        ffmpeg_path = None

    if mscore_path:
        st.success(f"✅ MuseScore found")
    else:
        st.warning("⚠️ MuseScore not found – sheet music will use piano-roll fallback")

    if lilypond_path:
        st.success(f"✅ LilyPond found")
    else:
        st.info("ℹ️ LilyPond not found (optional)")

# ─────────────────────────────────────────────────────────────────────────────
# Build configs from sidebar
# ─────────────────────────────────────────────────────────────────────────────
_res_map = {
    "1280 × 720":  (1280, 720),
    "1920 × 1080": (1920, 1080),
    "854 × 480":   (854, 480),
}
vw, vh = _res_map[res_choice]

video_cfg = VideoConfig(
    video_width=vw,
    video_height=vh,
    fps=fps,
    fall_speed=float(fall_speed),
    key_range_min=int(key_lo),
    key_range_max=int(key_hi),
)

pipe_cfg = PipelineConfig(
    max_duration=float(max_dur_s),
    transcription_mode="polyphonic" if "polyphonic" in trans_mode else "monophonic",
    onset_threshold=onset_thr,
    frame_threshold=frame_thr,
    min_note_ms=float(min_note_ms),
    quantize=quantize,
    video=video_cfg,
    score=ScoreConfig(quantize=quantize),
)

# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────
st.title("🎹 Piano Tutor")
st.markdown(
    "Upload a piano audio clip **or** paste a URL (YouTube, SoundCloud, direct link). "
    "The app will transcribe the audio, generate sheet music, and produce a "
    "falling-notes tutorial video."
)

# Input method
input_tab, url_tab = st.tabs(["📁 Upload file", "🔗 URL"])

source       = None
source_name  = "audio"
is_url       = False

with input_tab:
    uploaded = st.file_uploader(
        "Drag & drop audio file",
        type=["mp3", "wav", "flac", "ogg", "m4a", "aac", "opus", "webm", "aif", "aiff"],
        accept_multiple_files=False,
    )
    if uploaded is not None:
        source      = uploaded.read()
        source_name = uploaded.name
        is_url      = False
        st.audio(source, format=f"audio/{Path(uploaded.name).suffix.lstrip('.')}")

with url_tab:
    url_input = st.text_input(
        "Paste a URL (YouTube, SoundCloud, or direct audio link)",
        placeholder="https://www.youtube.com/watch?v=…",
    )
    if url_input.strip():
        source      = url_input.strip()
        source_name = "url_audio.wav"
        is_url      = True

    with st.expander("⚠️ YouTube 403 / blocked?  Read this first"):
        st.markdown("""
**YouTube now blocks many yt-dlp requests** without a signed-in browser session.
The app tries several bypass strategies automatically, but if all fail:

| Fix | How |
|-----|-----|
| **Use your browser cookies** | Open Chrome/Firefox, log into YouTube *(not incognito)*, then click Process – the app reads your cookies automatically |
| **Download manually** | Use [yt-dlp](https://github.com/yt-dlp/yt-dlp) on the command line: `yt-dlp -x --audio-format mp3 <URL>`, then upload the file |
| **Use a direct MP3/WAV link** | SoundCloud, archive.org, or any `.mp3` / `.wav` URL works without cookies |
| **Use a local file** | Switch to the **Upload file** tab |

> The Python 3.8 deprecation warning from yt-dlp is cosmetic – upgrade to Python ≥ 3.9 to silence it.
        """)


# ── Process button ────────────────────────────────────────────────────────────
st.divider()
col_btn, col_hint = st.columns([1, 4])
with col_btn:
    process_btn = st.button("▶ Process", type="primary", disabled=(source is None or ffmpeg_path is None))
with col_hint:
    if source is None:
        st.info("Provide an audio source above, then click **Process**.")
    elif ffmpeg_path is None:
        st.error("ffmpeg is required. Install it and restart the app.")

# ── Session state ─────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None

# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────
if process_btn and source is not None:
    st.session_state.result = None   # reset previous result

    progress_bar  = st.progress(0.0)
    status_text   = st.empty()
    step_statuses = [st.empty() for _ in _STEPS]

    def _progress_cb(step: int, total: int, fraction: float, message: str) -> None:
        progress_bar.progress(min(fraction, 1.0))
        label = message or (f"Step {step + 1}/{total}: {_STEPS[step]}" if step < len(_STEPS) else "")
        status_text.markdown(f"**{label}**")
        for i, ph in enumerate(step_statuses):
            if i < step:
                ph.markdown(f"✅ {_STEPS[i]}")
            elif i == step:
                ph.markdown(f"⏳ {_STEPS[i]} …")
            else:
                ph.markdown(f"⬜ {_STEPS[i]}")

    t_start = time.perf_counter()
    pipeline_error: Optional[Exception] = None
    result = None
    with st.spinner("Running pipeline …"):
        try:
            result = run_pipeline(
                source=source,
                source_name=source_name,
                is_url=is_url,
                cfg=pipe_cfg,
                progress_cb=_progress_cb,
            )
        except Exception as exc:
            pipeline_error = exc

    elapsed = time.perf_counter() - t_start
    progress_bar.progress(1.0)
    for ph in step_statuses:
        ph.empty()

    if pipeline_error is not None:
        status_text.error("Pipeline failed")
        err_msg = str(pipeline_error)
        st.error(f"**Error:** {err_msg}")
        if "403" in err_msg or "download strategies failed" in err_msg:
            st.info(
                "**YouTube 403 fix:** Open Chrome/Firefox, sign into YouTube "
                "(not incognito), then try again — the app will use your browser cookies.\n\n"
                "Alternatively, download the file manually with yt-dlp and use the Upload tab."
            )
        st.stop()

    status_text.success(f"Done in {elapsed:.1f}s")
    st.session_state.result = result

    if result and result.errors:
        for err in result.errors:
            # Non-fatal errors (sheet music, video) – shown as warnings
            st.warning(f"⚠️ {err}")

# ─────────────────────────────────────────────────────────────────────────────
# Display results
# ─────────────────────────────────────────────────────────────────────────────
result = st.session_state.get("result")
if result is not None and result.ok:
    st.divider()
    st.subheader("Results")

    # ── Timing summary ─────────────────────────────────────────────────────
    with st.expander("⏱ Timing breakdown"):
        t = result.timings
        cols = st.columns(5)
        for col, (label, key) in zip(
            cols,
            [("Total", "total"), ("Decode", "decode"),
             ("Transcribe", "transcribe"), ("Score", "score"), ("Video", "video")],
        ):
            col.metric(label, f"{t.get(key, 0):.1f}s")

    # ── Downloads ──────────────────────────────────────────────────────────
    dl_cols = st.columns(3)

    with dl_cols[0]:
        st.markdown("**MIDI**")
        if result.midi_path and result.midi_path.exists():
            st.download_button(
                "⬇ Download MIDI",
                data=result.midi_path.read_bytes(),
                file_name=result.midi_path.name,
                mime="audio/midi",
            )

    with dl_cols[1]:
        st.markdown("**Sheet Music**")
        if result.pdf_path and result.pdf_path.exists():
            st.download_button(
                "⬇ Download PDF",
                data=result.pdf_path.read_bytes(),
                file_name="score.pdf",
                mime="application/pdf",
            )
        else:
            st.caption("PDF unavailable (requires MuseScore/LilyPond)")

    with dl_cols[2]:
        st.markdown("**Tutorial Video**")
        if result.video_path and result.video_path.exists():
            st.download_button(
                "⬇ Download MP4",
                data=result.video_path.read_bytes(),
                file_name="piano_tutorial.mp4",
                mime="video/mp4",
            )

    st.divider()

    # ── Score preview ──────────────────────────────────────────────────────
    col_score, col_video = st.columns([1, 1])

    with col_score:
        st.markdown("### 📄 Sheet Music Preview")
        if result.png_paths:
            for i, png in enumerate(result.png_paths[:4]):   # show up to 4 pages
                if png.exists():
                    st.image(str(png), caption=f"Page {i + 1}", use_container_width=True)
        else:
            st.info("No sheet-music preview available.")

    with col_video:
        st.markdown("### 🎬 Tutorial Video")
        if result.video_path and result.video_path.exists():
            st.video(str(result.video_path))
        else:
            st.info("Video rendering failed or not yet complete.")

    # ── MIDI playback ──────────────────────────────────────────────────────
    if result.wav_path and result.wav_path.exists():
        st.divider()
        st.markdown("### 🎧 Original Audio (normalised WAV)")
        st.audio(str(result.wav_path))

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Piano Tutor · Open-source · "
    "[GitHub](https://github.com/your-org/piano-tutor) · "
    "Stack: Basic Pitch · music21 · MuseScore · OpenCV · ffmpeg · Streamlit"
)
