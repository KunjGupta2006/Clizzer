from pathlib import Path
from typing import Dict
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core Providers
    LLM_PROVIDER: str = "groq"
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_FALLBACK_MODEL: str = "llama-3.1-8b-instant"
    GROQ_RPM: int = 20

    HF_TOKEN: str = ""
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    GEMINI_API_KEY: str = ""

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = Path("data")
    CACHE_DIR: Path = Path("data/cache")
    JOBS_DIR: Path = Path("data/jobs")
    OUT_DIR: Path = Path("data/out")
    DB_PATH: Path = Path("data/jobs.db")

    # Modality Presets
    # Format: {"audio": float, "chat": float, "text": float, "visual": float, "face": float}
    WEIGHT_PRESETS: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "auto": {"audio": 0.25, "chat": 0.25, "text": 0.25, "visual": 0.15, "face": 0.10},
            "hype": {"audio": 0.30, "chat": 0.35, "text": 0.10, "visual": 0.15, "face": 0.10},
            "funny": {"audio": 0.25, "chat": 0.30, "text": 0.25, "visual": 0.10, "face": 0.10},
            "informational": {"audio": 0.15, "chat": 0.10, "text": 0.50, "visual": 0.15, "face": 0.10},
        }
    )

    def init_directories(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self.OUT_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.init_directories()
