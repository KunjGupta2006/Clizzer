# Clizzer: Autonomous Multimodal AI Video Clipper

**Clizzer** is an autonomous AI video clipper designed to transform long-form video content (YouTube livestreams, Twitch VODs, uploaded gaming videos, GTA 6 streams, reactions, and trailers) into viral vertical shorts (9:16, 1:1, or 16:9).

Clizzer fuses signals across 5 multimodal streams:
1. **Audio**: RMS energy, spectral flux, pitch variance, and AudioSet sound events (Transformers AST).
2. **Speech & Diarization**: Word-level ASR with `faster-whisper`, speaker segmentation with `pyannote.audio` (ECAPA fallback).
3. **Vision & Face**: PySceneDetect boundary cuts, InsightFace Buffalo_L & MediaPipe FaceMesh for speaker tracking, OpenCLIP zero-shot viral scoring, and RapidOCR.
4. **Live Chat**: Chat velocity spikes, sentiment scoring (VADER + emoji extension), hype/laughter/fail lexicons, and chat-to-video delay compensation.
5. **Semantic Text & LLM**: Boundary-aware topic segmentation, LLM narrative hooks and quality scoring (Groq LLaMA 3.3 / LLaMA 3.1, Ollama, or heuristic fallback).

Rendered outputs feature smart face-centered vertical reframing, word-highlight ASS subtitles, and EBU R128 (-14 LUFS) broadcast loudness normalization.

---

## 💻 Is My Machine Supported? (No GPU Required!)

> [!NOTE]
> **YES! Your machine is 100% supported even if you do not have a dedicated GPU (e.g. no NVIDIA graphics card).**
> Clizzer was engineered from day one with a **CPU-first, zero-GPU fallback architecture**.

### How Clizzer Runs on a Standard CPU Machine:
- **Speech Transcription (ASR)**: Uses `faster-whisper` quantized to `int8` on CPU. It runs smoothly on modern dual-core or quad-core Intel/AMD CPUs.
- **Video Rendering & Subtitles**: FFmpeg automatically detects whether an NVIDIA GPU is present. When no GPU is found, it automatically uses CPU-accelerated `libx264` (`-preset veryfast -crf 19`), which is universally compatible with every machine.
- **Face Tracking & Computer Vision**: InsightFace, MediaPipe FaceMesh, and PySceneDetect run natively using the CPU execution provider in ONNX Runtime.
- **LLM Reasoning & Viral Hooks**: Uses the **Groq Cloud API** (free tier at [console.groq.com](https://console.groq.com)), meaning large 70B models run on Groq's high-speed cloud chips rather than consuming your local RAM/GPU. If no internet or API key is available, it gracefully falls back to built-in heuristic scoring.
- **Audio Analysis**: Librosa energy, flux, and pitch extraction run purely in NumPy/SciPy on CPU.

### Minimum vs. Recommended Hardware Specs:
| Component | Minimum (No GPU) | Recommended (With GPU) |
| :--- | :--- | :--- |
| **Processor** | 4-core Intel Core i5 / AMD Ryzen 5 or Apple M1/M2/M3 | 8-core CPU or Apple Silicon |
| **RAM** | 8 GB RAM | 16 GB+ RAM |
| **GPU** | **None (Pure CPU Mode)** | NVIDIA GPU (RTX 2060+ with 6GB+ VRAM) |
| **Storage** | 5 GB free disk space | 20 GB free disk space (for large video caches) |
| **OS** | Windows 10/11, macOS 12+, or Linux (Ubuntu 20.04+) | Windows 10/11 or Ubuntu 22.04+ |

---

## 📦 Step-by-Step Installation Guides

Choose the guide matching your operating system:

<details open>
<summary><b>🪟 Windows Installation Guide (10 / 11)</b></summary>

### 1. Install System Dependencies

Open **PowerShell** as Administrator or use your favorite package manager:

#### Using Windows Package Manager (`winget` - Built into Windows 11 & 10):
```powershell
# Install Python 3.11
winget install Python.Python.3.11

# Install Node.js (LTS version 20)
winget install OpenJS.NodeJS.LTS

# Install FFmpeg (essential for video processing)
winget install Gyan.FFmpeg

# Install Git
winget install Git.Git
```
*Note: After installing with `winget`, close and reopen your PowerShell terminal so your `PATH` updates.*

#### Or install manually:
- [Python 3.11 Windows Installer](https://www.python.org/downloads/release/python-3119/) (⚠️ **Crucial**: Check the box **"Add python.exe to PATH"** during installation).
- [Node.js LTS Installer](https://nodejs.org/).
- [FFmpeg Windows Builds](https://www.gyan.dev/ffmpeg/builds/) (Extract and add the `bin/` folder to your System Environment `PATH`).

#### Verify installations:
```powershell
python --version   # Should show Python 3.11.x or 3.12.x
node --version     # Should show v18.x or v20.x
npm --version      # Should show 9.x or 10.x
ffmpeg -version    # Should print FFmpeg version info
```

### 2. Install `uv` (Ultra-fast Python Package Manager)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
*(Alternatively, you can just use `pip install uv`).*

### 3. Clone Repository & Install Dependencies
```powershell
# Clone the repository
git clone https://github.com/your-org/clizzer.git
cd clizzer

# Install Python backend dependencies in virtual environment
uv sync

# Or with standard pip:
# python -m venv .venv
# .venv\Scripts\activate
# pip install -e .

# Install Frontend dependencies
cd web
npm install
cd ..
```
</details>

<details>
<summary><b>🍎 macOS Installation Guide (Apple Silicon M1/M2/M3 & Intel)</b></summary>

### 1. Install Dependencies with Homebrew

If you don't have Homebrew installed, install it from [brew.sh](https://brew.sh):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install Python, Node.js, and FFmpeg:
```bash
brew install python@3.11 node ffmpeg git uv
```

Verify your installation:
```bash
python3 --version
node -v
ffmpeg -version
```

### 2. Clone Repository & Install Dependencies
```bash
git clone https://github.com/your-org/clizzer.git
cd clizzer

# Install Python backend dependencies
uv sync

# Install Frontend dependencies
cd web
npm install
cd ..
```
</details>

<details>
<summary><b>🐧 Linux Installation Guide (Ubuntu / Debian / Mint)</b></summary>

### 1. Install System Packages
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip nodejs npm ffmpeg git curl

# Install uv (recommended for ultra-fast builds)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 2. Clone Repository & Install Dependencies
```bash
git clone https://github.com/your-org/clizzer.git
cd clizzer

# Install Python backend dependencies
uv sync

# Install Frontend dependencies
cd web
npm install
cd ..
```
</details>

---

## ⚡ Quick Start: How to Start the Project

Once the installation above is complete, you can launch Clizzer in **two ways**:

### Option A: One-Command Start with Docker Compose (Recommended)

If you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed:

```bash
# 1. Copy the environment file (defaults work out of the box with zero keys)
cp .env.example .env

# 2. Build and launch all services (API + Web Frontend)
docker compose up --build
```

Once started:
- 🌐 **Web Dashboard**: [http://localhost:3000](http://localhost:3000)
- 🔌 **Backend REST API**: [http://localhost:8000](http://localhost:8000)
- 📚 **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

To run detached in background:
```bash
docker compose up -d
# View logs at any time:
docker compose logs -f
# Stop containers:
docker compose down
```

---

### Option B: Local Development Setup (Two Terminal Windows)

#### Terminal 1: Launch Backend API (FastAPI)
```bash
# In the project root directory:
cp .env.example .env

# Start FastAPI server on port 8000 with hot-reload:
uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# (Or if using standard virtual environment:)
# .venv\Scripts\activate   # Windows
# source .venv/bin/activate # Mac/Linux
# uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Terminal 2: Launch Frontend Dashboard (Next.js)
```bash
# In the project root directory:
cd web

# Start Next.js development server on port 3000:
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your web browser!

---

## 🎮 How to Use Clizzer

You can interact with Clizzer through the **Web Dashboard**, the **REST API**, or directly via **cURL**.

### Method 1: Using the Web Dashboard (Recommended)

1. **Open the App**:
   Navigate to [http://localhost:3000](http://localhost:3000) in your web browser.

2. **Submit a Video**:
   You have two ways to provide video input:
   - **YouTube / Twitch Link**: Paste the URL (e.g. `https://www.youtube.com/watch?v=...` or Twitch VOD). Clizzer will automatically download the media and replay chat.
   - **Upload Local File**: Toggle to **"File Upload"** and upload any `.mp4`, `.mkv`, `.webm`, or `.mov` video file.

3. **Configure Clipping Parameters**:
   - **Clips Count**: Choose between 1 and 30 clips (default: 5).
   - **Target Duration**: Set minimum (e.g., 20s) and maximum (e.g., 60s) duration per clip.
   - **Aspect Ratio**:
     - `9:16 (Vertical Short)`: Optimized for TikTok, YouTube Shorts, and Instagram Reels (features smart face tracking and letterboxing).
     - `1:1 (Square)`: Optimized for Instagram and Twitter feeds.
     - `16:9 (Landscape)`: Retains original aspect ratio.
   - **Optimization Mode**:
     - `Auto`: Balanced multimodal weighting across audio, chat, visual, and speech.
     - `Hype`: Amplifies chat velocity, audio loudness spikes, and high-energy gameplay.
     - `Funny`: Emphasizes laughter lexicons, chat emote explosions (`LMAO`, `KEKW`), and speech humor.
     - `Informational`: Focuses on speech clarity, topic segmentation, and coherent explanations.
   - **Keywords (Optional)**: Comma-separated topics to prioritize (e.g., `GTA 6, heist, stunt, Lucia, police chase`).
   - **Animated Captions**: Toggle word-by-word highlighted ASS subtitles.
   - **Rights Confirmation**: Check the required copyright/permission confirmation box.

4. **Monitor Real-Time Progress**:
   Click **"Generate Viral Clips"**. You will be redirected to the live Job Dashboard (`/jobs/[id]`) with a live WebSocket connection showing:
   - **Progress Bar & Elapsed Timers**.
   - **Active Pipeline Stage**: `Downloading` &rarr; `Transcribing` &rarr; `Analyzing` &rarr; `Ranking` &rarr; `Rendering` &rarr; `Done`.

5. **Review and Export Generated Clips**:
   - **Interactive Multimodal Radar / Score Curve**: Inspect the 1Hz fused score curve alongside audio, chat, text, visual, and face metrics.
   - **Clip Player**: Watch rendered 1080x1920 clips with auto-reframe, word captions, and loudness normalization.
   - **Fine-Tune Boundaries**: Adjust clip start/end timestamps using the interactive trimming handles.
   - **Metadata & Copy**: Review AI-generated titles (under 70 characters), viral hook sentences, and hashtags (`#GTA6`, `#Gaming`, etc.).
   - **Download**: One-click download of `.mp4` video files and `.srt` subtitle files.

---

### Method 2: Using the REST API & cURL

You can integrate Clizzer into your own scripts, bots, or pipelines using standard HTTP calls:

#### 1. Submit a Video URL Job

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "num_clips": 3,
    "min_len": 20,
    "max_len": 50,
    "aspect": "9:16",
    "mode": "hype",
    "keywords": ["epic", "reaction", "win"],
    "captions": true,
    "permission_confirmed": true
  }'
```

**Response**:
```json
{
  "id": "e4c2b9a1-8d2b-426b-95af-1b8389a9f2ce",
  "status": "queued",
  "progress": 0.0,
  "stage": "queued",
  "created_at": "2026-11-19T18:30:00Z"
}
```

#### 2. Submit a Local File Upload Job

```bash
curl -X POST http://localhost:8000/jobs \
  -F "file=@my_stream_recording.mp4" \
  -F "num_clips=5" \
  -F "aspect=9:16" \
  -F "mode=auto" \
  -F "captions=true" \
  -F "permission_confirmed=true"
```

#### 3. Poll Job Status

```bash
curl http://localhost:8000/jobs/e4c2b9a1-8d2b-426b-95af-1b8389a9f2ce
```

**Response (In Progress / Completed)**:
```json
{
  "id": "e4c2b9a1-8d2b-426b-95af-1b8389a9f2ce",
  "status": "done",
  "progress": 1.0,
  "stage": "done",
  "timings": {
    "ingest": 4.2,
    "transcribe": 8.1,
    "analyze": 12.3,
    "rank": 1.5,
    "render": 14.7
  }
}
```

#### 4. Retrieve Ranked Clips

```bash
curl http://localhost:8000/jobs/e4c2b9a1-8d2b-426b-95af-1b8389a9f2ce/clips
```

#### 5. Download Clip File Directly

```bash
curl -O http://localhost:8000/clips/clip_01/download
```

#### 6. Stream Live WebSocket Events

Connect to `ws://localhost:8000/ws/jobs/{id}` for live progress JSON frames:
```json
{"type": "progress", "stage": "analyzing", "progress": 0.45}
```

---

## 🔑 API Keys & External Services (Optional)

Clizzer is designed to be **100% functional offline with zero keys**. Connecting external services unlocks advanced LLM narrative understanding and deep neural speaker diarization:

### 1. Groq Cloud API (Recommended LLM)
Groq provides fast inference on LLaMA 3.3 and 3.1 models with a generous free tier:
1. Sign up for a free account at [console.groq.com](https://console.groq.com).
2. Go to **API Keys** &rarr; **Create API Key**.
3. Set the key in your `.env` file:
   ```env
   LLM_PROVIDER=groq
   GROQ_API_KEY=gsk_your_groq_api_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   GROQ_RPM=20
   ```
   *(If `llama-3.3-70b-versatile` reaches capacity limits, Clizzer automatically fails over to `llama-3.1-8b-instant` or heuristic text scoring without crashing).*

### 2. Hugging Face Token (Speaker Diarization)
Clizzer utilizes `pyannote.audio` 3.x for multi-speaker diarization:
1. Create a free account at [huggingface.co](https://huggingface.co).
2. Generate a Read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Accept the user conditions on the model pages:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Add your token to `.env`:
   ```env
   HF_TOKEN=hf_your_hugging_face_token_here
   ```
   *(If `HF_TOKEN` is unset or access is not granted, Clizzer logs a structured notice and cleanly falls back to SpeechBrain ECAPA / single-speaker mode).*

### 3. Ollama (100% Local LLM Alternative)
If you prefer running a local LLM without cloud services:
1. Install Ollama from [ollama.com](https://ollama.com).
2. Download a model: `ollama pull llama3.2`
3. Configure in `.env`:
   ```env
   LLM_PROVIDER=ollama
   OLLAMA_HOST=http://localhost:11434
   OLLAMA_MODEL=llama3.2
   ```

---

## 🧪 Verification & Testing

Clizzer includes a comprehensive test suite with 100% synthetic fixtures that run offline without external downloads:

```bash
# Run backend test suite (46 tests, 0 skipped, 0 failures, 0 API keys required)
uv run pytest -q

# Run frontend production build
cd web
npm run build
cd ..

# Run multimodal evaluation harness & ablation study
python eval/harness.py
```

### Evaluation Output Example:
```text
================================================================================
CLIZZER MULTIMODAL EVALUATION HARNESS
Dataset: 300s synthetic stream with 4 planted viral highlights
Hit criterion: Predicted start within +/- 5.0s of ground truth
================================================================================
Experiment                       | Precision@4  | Recall@4   | Delta P   | Delta R  
--------------------------------------------------------------------------------
Full System (Baseline)           |    100.00% |  100.00% |   +0.00% |   +0.00%
Ablation: No Audio               |    100.00% |  100.00% |   +0.00% |   +0.00%
Ablation: No Chat                |    100.00% |  100.00% |   +0.00% |   +0.00%
Ablation: No Text                |    100.00% |  100.00% |   +0.00% |   +0.00%
Ablation: No Visual              |    100.00% |  100.00% |   +0.00% |   +0.00%
Ablation: No Face                |    100.00% |  100.00% |   +0.00% |   +0.00%
================================================================================
```

---

## 📁 Architecture & Directory Layout

```
clizzer/
  DECISIONS.md          # Architectural decisions log (D001-D008)
  PROGRESS.md           # Implementation phase tracking (P0-P9)
  pyproject.toml        # Python project configuration and dependencies
  .env.example          # Environment variables template
  Dockerfile            # Backend container specification
  docker-compose.yml    # Dual-service orchestrator (API + Web)
  backend/app/
    main.py             # FastAPI entrypoint and lifespan handlers
    config.py           # Pydantic settings loading from .env
    db.py               # SQLModel database session engine
    models.py           # Pydantic & SQLModel contracts
    errors.py           # Typed domain exception hierarchy
    logging.py          # Structlog structured JSON logger
    api/                # API route handlers (/jobs, /clips, /ws)
    core/               # Orchestrator, persistent cache, progress reporter
    ingest/             # Video downloader, FFmpeg utils, chat fetcher, chunker
    audio/              # ASR (whisper), diarization, audio features, events
    vision/             # PySceneDetect, face tracking, OpenCLIP, OCR
    chat/               # Chat features, sentiment/lexicons, lag alignment
    text/               # Semantic segmentation, LLM client, heuristic scoring
    rank/               # Fusion z-score, candidate windowing, NMS/MMR dedup
    render/             # Reframe 9:16, ASS subtitles, -14 LUFS loudnorm
    eval/harness.py     # Benchmark dataset, precision/recall evaluation
  backend/tests/        # Comprehensive unit & E2E integration test suite
  web/                  # Next.js 14 App Router frontend
    app/                # Page routes (dashboard and job detail view)
    components/         # SubmitForm, ProgressTimeline, ScoreCurve, ClipCard, TrimHandles
    lib/api.ts          # Typed API client and WebSocket handlers
  eval/harness.py       # Evaluation runner wrapper
```

---

## 📄 License

MIT License. Designed and built autonomously with the Google DeepMind Antigravity framework.
