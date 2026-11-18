"""Configuration settings for Clizzer backend.

Provides a frozen dataclass with default pipeline parameters, model options,
scoring weights, audio/video specifications, and helper methods to load
overrides from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self


@dataclass(frozen=True)
class Config:
    """Clizzer backend configuration settings.

    Attributes:
        TEMP_DIR: Directory for intermediate scratch files.
        OUTPUT_DIR: Directory for final exported clip artifacts.
        MODELS_DIR: Directory for caching AI/ML models.
        WHISPER_MODEL_SIZE: Model size identifier for faster-whisper (default 'tiny').
        WHISPER_COMPUTE_TYPE: Quantization type for inference (default 'int8').
        WHISPER_DEVICE: Hardware device target for inference (default 'cpu').
        AUDIO_WEIGHT: Scoring weight for audio excitation and onsets.
        MOTION_WEIGHT: Scoring weight for visual frame motion.
        FACE_WEIGHT: Scoring weight for facial engagement and reactions.
        SENTIMENT_WEIGHT: Scoring weight for speech transcript sentiment.
        CROWD_WEIGHT: Scoring weight for crowd cheering / laughter cues.
        MIN_CLIP_DURATION: Minimum allowed clip duration in seconds.
        MAX_CLIP_DURATION: Maximum allowed clip duration in seconds.
        NUM_CLIPS: Target number of highlighted clips to generate.
        CLIP_PADDING: Additional padding in seconds before/after clip boundaries.
        OUTPUT_FORMAT: Target video container format extension.
        VERTICAL_RESOLUTION: Dimensions (width, height) for vertical format (9:16).
        SAMPLE_RATE: Audio sampling frequency in Hz.
        HOP_LENGTH: Audio feature extraction hop length in samples.
        FRAME_SAMPLE_INTERVAL: Frame subsampling interval in seconds for visual analysis.
    """

    TEMP_DIR: Path = Path("temp")
    OUTPUT_DIR: Path = Path("output")
    MODELS_DIR: Path = Path("models_cache")

    WHISPER_MODEL_SIZE: str = "tiny"
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_DEVICE: str = "cpu"

    AUDIO_WEIGHT: float = 0.30
    MOTION_WEIGHT: float = 0.20
    FACE_WEIGHT: float = 0.20
    SENTIMENT_WEIGHT: float = 0.15
    CROWD_WEIGHT: float = 0.15

    MIN_CLIP_DURATION: int = 15
    MAX_CLIP_DURATION: int = 60
    NUM_CLIPS: int = 4
    CLIP_PADDING: float = 2.0

    OUTPUT_FORMAT: str = "mp4"
    VERTICAL_RESOLUTION: tuple[int, int] = (1080, 1920)

    SAMPLE_RATE: int = 22050
    HOP_LENGTH: int = 512
    FRAME_SAMPLE_INTERVAL: float = 0.5

    def ensure_directories(self) -> None:
        """Create configured storage directories on disk if they do not exist."""
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> Self:
        """Load configuration parameters from environment variables with fallbacks.

        Returns:
            Config: Fully initialized frozen configuration instance.
        """
        def _get_str(name: str, fallback: str) -> str:
            return os.getenv(name, fallback).strip()

        def _get_path(name: str, fallback: str) -> Path:
            val = os.getenv(name)
            return Path(val.strip()) if val and val.strip() else Path(fallback)

        def _get_int(name: str, fallback: int) -> int:
            val = os.getenv(name)
            if val is None or not val.strip():
                return fallback
            try:
                return int(val.strip())
            except ValueError:
                return fallback

        def _get_float(name: str, fallback: float) -> float:
            val = os.getenv(name)
            if val is None or not val.strip():
                return fallback
            try:
                return float(val.strip())
            except ValueError:
                return fallback

        def _get_resolution(name: str, fallback: tuple[int, int]) -> tuple[int, int]:
            val = os.getenv(name)
            if not val or not val.strip():
                return fallback
            cleaned = val.strip().lower().replace("x", ",").replace(":", ",")
            parts = [p.strip() for p in cleaned.split(",") if p.strip()]
            if len(parts) == 2:
                try:
                    return (int(parts[0]), int(parts[1]))
                except ValueError:
                    return fallback
            return fallback

        return cls(
            TEMP_DIR=_get_path("TEMP_DIR", "temp"),
            OUTPUT_DIR=_get_path("OUTPUT_DIR", "output"),
            MODELS_DIR=_get_path("MODELS_DIR", "models_cache"),
            WHISPER_MODEL_SIZE=_get_str("WHISPER_MODEL_SIZE", "tiny"),
            WHISPER_COMPUTE_TYPE=_get_str("WHISPER_COMPUTE_TYPE", "int8"),
            WHISPER_DEVICE=_get_str("WHISPER_DEVICE", "cpu"),
            AUDIO_WEIGHT=_get_float("AUDIO_WEIGHT", 0.30),
            MOTION_WEIGHT=_get_float("MOTION_WEIGHT", 0.20),
            FACE_WEIGHT=_get_float("FACE_WEIGHT", 0.20),
            SENTIMENT_WEIGHT=_get_float("SENTIMENT_WEIGHT", 0.15),
            CROWD_WEIGHT=_get_float("CROWD_WEIGHT", 0.15),
            MIN_CLIP_DURATION=_get_int("MIN_CLIP_DURATION", 15),
            MAX_CLIP_DURATION=_get_int("MAX_CLIP_DURATION", 60),
            NUM_CLIPS=_get_int("NUM_CLIPS", 4),
            CLIP_PADDING=_get_float("CLIP_PADDING", 2.0),
            OUTPUT_FORMAT=_get_str("OUTPUT_FORMAT", "mp4").lstrip("."),
            VERTICAL_RESOLUTION=_get_resolution("VERTICAL_RESOLUTION", (1080, 1920)),
            SAMPLE_RATE=_get_int("SAMPLE_RATE", 22050),
            HOP_LENGTH=_get_int("HOP_LENGTH", 512),
            FRAME_SAMPLE_INTERVAL=_get_float("FRAME_SAMPLE_INTERVAL", 0.5),
        )


# Default global instance initialized from environment variables
config = Config.from_env()
