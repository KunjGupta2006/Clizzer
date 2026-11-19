from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings
from backend.app.db import init_db
from backend.app.logging import logger
from backend.app.api.jobs import router as jobs_router
from backend.app.api.clips import router as clips_router
from backend.app.api.ws import router as ws_router

app = FastAPI(
    title="Clizzer API",
    version="0.1.0",
    description="Multimodal AI Video Clipper",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router)
app.include_router(clips_router)
app.include_router(ws_router)

# Initialize tables immediately
init_db()


@app.on_event("startup")
def on_startup() -> None:
    settings.init_directories()
    init_db()
    logger.info("clizzer_startup_complete")


@app.get("/health")
def health() -> dict:
    # Reports live paths without exposing keys
    llm_status = "heuristic"
    if settings.GROQ_API_KEY:
        llm_status = "groq"
    elif settings.OLLAMA_HOST:
        llm_status = "ollama"

    diar_status = "pyannote" if settings.HF_TOKEN else "speechbrain"

    return {
        "status": "healthy",
        "llm": llm_status,
        "diarization": diar_status,
        "gpu": False,
    }
