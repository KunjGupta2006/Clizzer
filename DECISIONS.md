# CLIZZER ARCHITECTURAL & TECHNICAL DECISIONS

This document logs all key decisions made during autonomous implementation of the Clizzer system.

## D001: Project Layout and Module Organization
- Target backend directory is `backend/app/` with clean subpackages: `api/`, `core/`, `ingest/`, `audio/`, `vision/`, `chat/`, `text/`, `rank/`, `render/`, `eval/`.
- Tests are organized under `backend/tests/` matching each module + `test_e2e.py`.
- Frontend is placed in `web/` using Next.js 14 App Router, TypeScript, Tailwind CSS, and Recharts.

## D002: In-Process Job Execution and Storage
- SQLite via SQLModel is used for persistence in `data/jobs.db`.
- Background processing uses Python `concurrent.futures.ThreadPoolExecutor` within the FastAPI process. No external Redis or Celery dependency.

## D003: Model Fallback Philosophy
- Every modality model loader returns `object | None` on missing dependencies/tokens/weights and logs a structured warning without throwing uncaught exceptions.
- Missing modalities produce zero-vectors in `SeriesBundle` and have `available=False`. The fusion step renormalizes weights over available modalities so that clipping and ranking continue without interruption.
