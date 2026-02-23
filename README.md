# 🎹 Piano Tutor

An open-source Streamlit app that turns any piano audio clip into:

| Output | Format | Description |
|--------|--------|-------------|
| **MIDI** | `.mid` | Full polyphonic transcription |
| **Sheet music** | PDF + PNG | Treble + Bass grand staff |
| **Tutorial video** | MP4 | Falling notes + labeled piano keys + scrolling score overlay |

---

## Demo

Upload a 10–30 s piano recording **or** paste a YouTube/SoundCloud URL, click **Process**, and download your outputs.

---

## Architecture

```
Audio (upload / URL)
      │
      ▼  audio_io.py
   WAV (mono, 22 kHz)
      │
      ▼  transcribe.py
   MIDI  ──────────────────────────────────┐
      │                                    │
      ▼  midi_to_score.py                  ▼  render_video.py
 MusicXML → PDF + PNG           Falling-notes MP4 (+ audio mux)
      │                                    │
      └────────────────┬───────────────────┘
                       ▼
                 Streamlit UI (app.py)
```

### Open-source stack

| Concern | Library |
|---------|---------|
| Audio decode | `ffmpeg` (via subprocess) |
| URL download | `yt-dlp` |
| Audio analysis | `librosa` (pYIN fallback) |
| Polyphonic transcription | **Spotify Basic Pitch** |
| MIDI utilities | `pretty_midi` |
| Music notation | `music21` → MusicXML |
| Score rendering | **MuseScore** CLI → PDF/PNG *(LilyPond fallback; matplotlib piano-roll as last resort)* |
| Video frames | `numpy` + **OpenCV** |
| Video assembly | `ffmpeg` (rawvideo pipe → H.264) |
| Web UI | **Streamlit** |

---

## Local Setup

### 1 – Prerequisites

#### macOS
```bash
brew install ffmpeg
# Pick one (MuseScore recommended):
brew install --cask musescore      # MuseScore 4
# OR
brew install lilypond
```

#### Ubuntu / Debian
```bash
sudo apt update
sudo apt install ffmpeg musescore3   # or lilypond
```

#### Windows (WSL2 recommended)
Install ffmpeg and MuseScore manually from their official sites, then add them to PATH.

### 2 – Python environment
```bash
# Python 3.10+ required
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **Note:** `basic-pitch>=0.3` uses ONNX Runtime (lighter than TensorFlow).
> On Apple Silicon, `onnxruntime` installs the ARM build automatically.

### 3 – Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Quick test (CLI smoke test)

```bash
# Generate a short test MIDI and render the video without going through the UI:
python - <<'EOF'
from pathlib import Path
from src.pipeline import run_pipeline, PipelineConfig

# Use any .wav/.mp3 file you have, e.g. a 15-second piano clip:
result = run_pipeline(
    source=Path("my_clip.wav"),
    source_name="my_clip.wav",
    cfg=PipelineConfig(),
)
print("MIDI   :", result.midi_path)
print("PDF    :", result.pdf_path)
print("PNGs   :", result.png_paths)
print("Video  :", result.video_path)
print("Errors :", result.errors)
EOF
```

### Getting a sample audio file

Any short piano recording works. A few free options:

```bash
# 1. Download a free public-domain piano clip (Bach Minuet) via yt-dlp:
yt-dlp -x --audio-format wav -o sample.wav "https://www.youtube.com/watch?v=GRxofEmo3HA"

# 2. Generate a synthetic test clip with ffmpeg (pure sine tones – transcription demo):
ffmpeg -f lavfi \
  -i "sine=frequency=261.63:duration=0.5,sine=frequency=293.66:duration=0.5,\
sine=frequency=329.63:duration=0.5,sine=frequency=349.23:duration=0.5" \
  -ar 22050 -ac 1 sample_tones.wav

# 3. Use any .mp3/.wav file from your music library.
```

---

## Running tests

```bash
pytest tests/ -v
# With coverage:
pytest tests/ --cov=src --cov-report=term-missing
```

Tests cover:
- `test_utils.py` – note names, MIDI helpers, hashing
- `test_audio_io.py` – file decode, caching (requires `ffmpeg`)
- `test_transcribe.py` – pYIN fallback, tempo detection
- `test_render_video.py` – keyboard layout, frame generation
- `test_midi_to_score.py` – piano-roll fallback, caching

---

## Advanced Settings (UI sidebar)

| Setting | Description |
|---------|-------------|
| **Transcription mode** | *Polyphonic* (Basic Pitch, chords) / *Monophonic* (pYIN, single melody) |
| **Onset threshold** | 0.1–0.9; lower = more note detections, may add false positives |
| **Frame threshold** | Confidence threshold for note sustain |
| **Min note length (ms)** | Discard notes shorter than this |
| **Quantise score** | Snap to 16th-note grid for cleaner notation |
| **Resolution** | 854×480 / 1280×720 / 1920×1080 |
| **FPS** | 15–60; lower = faster render, choppier video |
| **Fall speed (px/s)** | How fast notes descend toward the keyboard |
| **Key range** | MIDI note numbers; default A0–C8 (full piano) |

---

## Deployment

### Option A – Streamlit Community Cloud (recommended for sharing)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. `packages.txt` installs `ffmpeg` automatically.
4. **Sheet music** rendering requires MuseScore which is not available on Streamlit Cloud by default → the app falls back to the **matplotlib piano-roll renderer** (still useful for learning).
5. MIDI and video outputs work fully in the cloud.

To enable MuseScore on Streamlit Cloud, uncomment `musescore3` in `packages.txt` (**may exceed build limits**).

### Option B – Docker (full functionality everywhere)

```dockerfile
# Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg musescore3 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t piano-tutor .
docker run -p 8501:8501 piano-tutor
```

Deploy to **Fly.io**, **Render**, or **Railway** using the Dockerfile above.

---

## Project structure

```
piano-tutor/
├── app.py                 # Streamlit entry point
├── src/
│   ├── __init__.py
│   ├── pipeline.py        # Orchestrates all steps, caching, progress
│   ├── audio_io.py        # Download (yt-dlp), decode (ffmpeg)
│   ├── transcribe.py      # Basic Pitch + pYIN fallback → MIDI
│   ├── midi_to_score.py   # music21 + MuseScore/LilyPond → PDF/PNG
│   ├── render_video.py    # OpenCV + ffmpeg → falling-notes MP4
│   └── utils.py           # Note names, hashing, tool discovery
├── tests/
│   ├── test_utils.py
│   ├── test_audio_io.py
│   ├── test_transcribe.py
│   ├── test_midi_to_score.py
│   └── test_render_video.py
├── assets/                # Static assets (currently empty)
├── cache/                 # Pipeline output cache (git-ignored)
├── .streamlit/
│   └── config.toml        # Theme + upload size
├── packages.txt           # System packages for Streamlit Cloud
├── requirements.txt
├── LICENSE                # MIT
└── README.md
```

---

## Caching

All intermediate outputs are cached under `cache/{audio_hash}/`:

```
cache/
└── a1b2c3d4e5f6a7b8/
    ├── a1b2c3d4e5f6a7b8_sr22050_mono.wav   # decoded audio
    ├── audio_basic_pitch.mid               # transcribed MIDI
    └── score/
        ├── score.xml                       # MusicXML
        ├── score.pdf                       # sheet music (if MuseScore found)
        ├── score_page-1.png                # page previews
        └── tutorial.mp4                   # falling-notes video
```

Re-running the same audio file skips all processed steps.

---

## Limitations & known issues

| Issue | Workaround |
|-------|------------|
| Polyphonic transcription accuracy | Basic Pitch works well for solo piano; complex arrangements may have errors |
| Score rendering requires MuseScore/LilyPond | Piano-roll PNG always available as fallback |
| Long audio (> 3 min) | Video rendering is slow on CPU; use ≤ 60 s for best experience |
| MuseScore 4 on some Linux headless systems | Try `xvfb-run mscore` or switch to LilyPond |
| yt-dlp age-restricted videos | Provide a direct audio file instead |

---

## Contributing

PRs welcome! Priority areas:
- Better tempo/meter detection (current: librosa beat tracker)
- Measure-accurate score scrolling (current: linear approximation)
- GPU acceleration for Basic Pitch inference
- WASM/browser-side rendering

---

## License

MIT – see [LICENSE](LICENSE).

Depends on open-source projects: Basic Pitch (Apache-2.0), music21 (BSD), MuseScore (GPL-2), LilyPond (GPL-2), librosa (ISC), OpenCV (Apache-2.0), ffmpeg (LGPL-2.1+), yt-dlp (Unlicense), Streamlit (Apache-2.0).
