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

## Key Features & Guarantees

- **Zero-Keys Offline Fallback Mode**: Clizzer runs completely offline without any API keys or GPU. Every modality gracefully degrades to built-in heuristic scoring if external services or models are absent.
- **Hardware Agnostic**: Automatic detection of CUDA / CPU and FFmpeg hardware acceleration (`h264_nvenc` with automatic fallback to CPU `libx264`).
- **In-Process Pipeline**: No Redis or external task brokers needed. Uses SQLite and Python thread workers.
- **Full-Stack Web App**: Next.js 14 App Router dashboard with live WebSocket progress tracking, interactive score curve analysis, and trimming controls.

---

## System Requirements

- **Python**: 3.11 or 3.12 (with `uv` or `pip`)
- **Node.js**: 18+ or 20+ (with `npm`)
- **FFmpeg & FFprobe**: Installed and available in PATH (or provided via `static-ffmpeg`)
- **Operating System**: Linux, macOS, or Windows 10/11

---

## Quickstart (Local Development)

### 1. Clone & Set Up Backend

```bash
# Clone the repository
git clone https://github.com/your-org/clizzer.git
cd clizzer

# Install Python dependencies using uv
uv sync

# Or using standard pip in a virtual environment:
# python -m venv .venv
# .venv\Scripts\activate  (Windows) or source .venv/bin/activate (Linux/Mac)
# pip install -e .
```

### 2. Configure Environment

Copy the example environment configuration:

```bash
cp .env.example .env
```

To run completely offline with zero API keys, no edits are necessary. Clizzer will automatically use its offline heuristic fallbacks.

### 3. Launch Backend Server

```bash
uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be available at `http://localhost:8000/docs`.

### 4. Launch Frontend (Web UI)

In a separate terminal:

```bash
cd web
npm install
npm run dev
```
Open `http://localhost:3000` in your browser.

---

## API Keys & External Services (Optional)

Clizzer delivers highest accuracy when connected to external AI services:

### 1. Groq Cloud API (Recommended LLM)
Groq provides ultra-fast LLM inference with a generous free tier.
1. Sign up for a free account at [console.groq.com](https://console.groq.com).
2. Go to **API Keys** -> **Create API Key**.
3. Set the key in your `.env` file:
   ```env
   LLM_PROVIDER=groq
   GROQ_API_KEY=gsk_your_groq_api_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   GROQ_RPM=20
   ```
   *(If `llama-3.3-70b-versatile` experiences rate limits or capacity constraints, Clizzer automatically fails over to `llama-3.1-8b-instant`).*

### 2. Hugging Face Token (Speaker Diarization)
Clizzer utilizes `pyannote.audio` 3.x for speaker diarization. PyAnnote models are gated on Hugging Face:
1. Create a free account at [huggingface.co](https://huggingface.co).
2. Generate a Read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Accept the user conditions on the model pages:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Add your token to `.env`:
   ```env
   HF_TOKEN=hf_your_hugging_face_token_here
   ```
   *(If `HF_TOKEN` is not set or access is not granted, Clizzer logs a structured notice and falls back to SpeechBrain ECAPA / single-speaker mode without crashing).*

### 3. Ollama (Local LLM Alternative)
If you prefer 100% local LLM inference:
1. Install Ollama from [ollama.com](https://ollama.com).
2. Pull a model: `ollama pull llama3.2`
3. Configure `.env`:
   ```env
   LLM_PROVIDER=ollama
   OLLAMA_HOST=http://localhost:11434
   OLLAMA_MODEL=llama3.2
   ```

---

## CPU vs. GPU Instructions

Clizzer automatically detects available hardware:

- **CUDA GPU Available**:
  - `faster-whisper` runs using `float16` on CUDA.
  - FFmpeg attempts hardware acceleration via `h264_nvenc`.
  - Torch models (AST audio events, OpenCLIP visual embeddings) utilize GPU VRAM with memory cleanup after each phase.
- **CPU Only**:
  - `faster-whisper` runs using quantized `int8` on CPU.
  - FFmpeg encodes via `libx264 -crf 19 -preset veryfast`.
  - Multiprocessing threads optimize for CPU core counts.

To force CPU execution even when a GPU is attached, set:
```env
CUDA_VISIBLE_DEVICES=""
```

---

## Docker Compose Deployment

To run the entire Clizzer stack (FastAPI Backend + Next.js Frontend) in isolated Docker containers:

```bash
# Build and start services
docker compose up --build

# Run in background (detached)
docker compose up -d

# View logs
docker compose logs -f

# Shut down services
docker compose down
```

Once running:
- **Frontend Dashboard**: `http://localhost:3000`
- **Backend API**: `http://localhost:8000`
- **Interactive OpenAPI Docs**: `http://localhost:8000/docs`

---

## Verification & Testing

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

## Architecture & Directory Layout

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

## License

MIT License. Designed and built autonomously with the Google DeepMind Antigravity framework.
