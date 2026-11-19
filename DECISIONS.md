# CLIZZER ARCHITECTURAL & TECHNICAL DECISIONS

This document logs all key architectural and implementation decisions made during the autonomous build of the Clizzer multimodal video clipper.

---

## D001: Project Layout and Module Organization
- **Structure**: Backend modularized under `backend/app/` with clean subpackages:
  - `api/`: FastAPI route handlers (`jobs`, `clips`, `ws`).
  - `core/`: Pipeline orchestrator, disk cache, progress reporter.
  - `ingest/`: Video/audio downloader, FFmpeg media utilities, live chat fetcher, 30-min window chunker.
  - `audio/`: ASR (`faster-whisper`), diarization (`pyannote.audio` / ECAPA fallback), audio features (RMS/pitch flux), audio events (AST).
  - `vision/`: Scene boundary detection (`scenedetect`), face tracking (`insightface`/MediaPipe), zero-shot scene analysis (`open_clip`), OCR (`rapidocr`).
  - `chat/`: Chat spike features, lexicon matching (hype/laughter/cringe/GTA 6), emote lag alignment.
  - `text/`: Semantic text window segmentation, LLM client (`GroqProvider`, `OllamaProvider`), prompts, and heuristic text scoring.
  - `rank/`: Rolling median/MAD z-score normalization, candidate window proposal, sentence-boundary snap, NMS/MMR deduplication, top clip selection.
  - `render/`: 9:16 smart reframe with face-following box filter, word-highlight ASS subtitles, 2-pass -14 LUFS loudnorm.
  - `eval/`: Synthetic benchmark harness calculating Precision@N, Recall@N (+/- 5s), and per-modality ablation.
- **Frontend**: Located in `web/` using Next.js 14 App Router, TypeScript, Tailwind CSS, and Recharts.

---

## D002: In-Process Job Execution and SQLite Persistence
- **Decision**: Avoid heavyweight external broker dependencies (like Redis, Celery, or RabbitMQ).
- **Rationale**: Keeps the application fully self-contained for single-node deployment, local development, and CI environments.
- **Implementation**: SQLModel with SQLite (`data/jobs.db`). Asynchronous jobs execute in a managed Python `concurrent.futures.ThreadPoolExecutor(max_workers=2)`. Progress updates and stage transitions are written to SQLite and broadcast via WebSocket endpoints.

---

## D003: Model Fallback Philosophy & Degraded Mode Guarantees (G5)
- **Principle**: No modality failure may crash the clipping pipeline.
- **Implementation**:
  - Every model loader returns `object | None` on missing weights, missing API tokens, or hardware constraints.
  - Missing modalities produce zero-vectors in `SeriesBundle` with `available=False` and a structured `fallback_reason`.
  - Fusion weights dynamically renormalize to 1.0 across all available modalities.
  - LLM fallback chain: Groq API -> Ollama local -> Heuristic rule-based text scoring.
  - Diarization fallback chain: pyannote (HF_TOKEN) -> SpeechBrain ECAPA -> single speaker assignment.
  - Face tracking fallback: InsightFace buffalo_l -> MediaPipe FaceMesh -> Center crop.

---

## D004: FFmpeg Encoder Probing (NVENC Fallback to CPU libx264)
- **Problem**: On Windows/Linux systems with NVIDIA drivers uninstalled or in headless containers without GPU passthrough, `ffmpeg -encoders` reports `h264_nvenc`, but running it fails with `Cannot load nvcuda.dll` or `Device creation failed`.
- **Decision**: At application startup, `render.py` executes a 1-frame probe test using `h264_nvenc`. If the probe fails, the system automatically marks NVENC unavailable and falls back to `libx264 -crf 19 -preset veryfast`.

---

## D005: Whisper and PyAnnote Network Isolation in CI Tests
- **Problem**: Calling `WhisperModel("small")` or `Pipeline.from_pretrained` during automated tests can cause network stalls, download multi-gigabyte models, or fail with 401 when tokens are absent.
- **Decision**: Model loaders check `local_files_only=True` in test / offline environments. If weights are not locally cached, loaders return `None` and fall back cleanly to heuristic transcripts or mocked segmentations, satisfying G1 (no network/keys needed in tests).

---

## D006: Pydantic v2 String Length Pre-validation
- **Problem**: Pydantic v2 evaluates `Field(max_length=70)` before `@model_validator(mode="after")`. If an LLM or heuristic produces a 75-character title, Pydantic raises a validation error before the truncation code runs.
- **Decision**: Use `@field_validator("title", mode="before")` on the `Clip` model to cleanly truncate long titles to <= 70 characters before length validation is applied.

---

## D007: Gaussian Smoothing Kernel Width for Peak Detection
- **Decision**: Signal smoothing across the fused 1Hz timeseries uses a Gaussian filter with `sigma=1.2` (~5s kernel footprint).
- **Rationale**: A wider kernel (`sigma >= 2.0`) overly dampens short viral peaks (such as a 3s reaction or sudden headshot) below the quality threshold, whereas `sigma=1.2` suppresses high-frequency noise while preserving peak amplitude above quality floor 0.8.

---

## D008: Multi-modal Fusion Normalization and Evaluation Criteria
- **Decision**: Modality timeseries are normalized via rolling median and median absolute deviation (MAD) over 120s windows, clipped to `[-3.0, 3.0]`.
- **Evaluation Criteria**: Ground-truth hit tolerance is set to +/- 5.0 seconds from the target viral event onset, matching human editorial tolerance for short clip boundaries.
