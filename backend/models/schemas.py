"""Pydantic v2 data models and schemas for Clizzer.

Defines validated data structures for pipeline inputs, intermediate feature representations,
hype scoring, candidate clip windows, and final pipeline outputs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator


class VideoInput(BaseModel):
    """Input payload representing video source to be processed.

    Requires at least one valid source: a remote URL or a local filesystem path.
    """

    url: str | None = Field(
        default=None,
        description="Remote video URL (e.g., YouTube link or direct video stream).",
    )
    file_path: str | None = Field(
        default=None,
        description="Local filesystem path to the video file.",
    )

    @model_validator(mode="after")
    def validate_at_least_one_source(self) -> Self:
        """Ensure that at least one of url or file_path is provided and non-empty."""
        has_url = bool(self.url and self.url.strip())
        has_file = bool(self.file_path and self.file_path.strip())
        if not has_url and not has_file:
            raise ValueError("At least one of 'url' or 'file_path' must be provided.")
        return self


class ProcessingStatus(BaseModel):
    """Status update representing pipeline stage and execution progress."""

    step: str = Field(
        description="Current pipeline processing step identifier or name.",
    )
    progress: float = Field(
        ge=0.0,
        le=1.0,
        description="Pipeline progress fraction clamped between 0.0 and 1.0.",
    )
    message: str = Field(
        description="Human-readable status or informational message.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the status update was recorded.",
    )


class AudioFeatures(BaseModel):
    """Extracted acoustic and temporal feature vectors across audio frames."""

    rms_energy: list[float] = Field(
        description="Root Mean Square energy values across audio analysis frames.",
    )
    spectral_centroid: list[float] = Field(
        description="Spectral centroid frequencies in Hz across frames.",
    )
    percussive_energy: list[float] = Field(
        description="Percussive audio component energy across frames.",
    )
    onsets: list[float] = Field(
        description="Timestamps in seconds of detected musical or speech onsets.",
    )
    timestamps: list[float] = Field(
        description="Time points in seconds corresponding to feature frames.",
    )


class SceneInfo(BaseModel):
    """Metadata describing a detected continuous visual scene."""

    start_time: float = Field(
        ge=0.0,
        description="Start timestamp of the scene segment in seconds.",
    )
    end_time: float = Field(
        ge=0.0,
        description="End timestamp of the scene segment in seconds.",
    )
    scene_index: int = Field(
        ge=0,
        description="Zero-based sequential index of the detected scene.",
    )


class FaceData(BaseModel):
    """Visual facial detection, tracking coordinates, and expressive reaction metrics."""

    bbox: tuple[int, int, int, int] = Field(
        description="Bounding box coordinates (x, y, width, height) of the detected face.",
    )
    center_x: float = Field(
        description="Horizontal center position of the face.",
    )
    center_y: float = Field(
        description="Vertical center position of the face.",
    )
    mouth_open_ratio: float = Field(
        description="Metric representing mouth openness extent.",
    )
    smile_ratio: float = Field(
        description="Metric representing smiling or expressive excitement.",
    )
    timestamp: float = Field(
        ge=0.0,
        description="Timestamp in seconds for the analyzed frame.",
    )


class TranscriptSegment(BaseModel):
    """Transcribed speech segment with timing, word metadata, and sentiment score."""

    text: str = Field(
        description="Transcribed text content of the speech segment.",
    )
    start: float = Field(
        ge=0.0,
        description="Start timestamp of the segment in seconds.",
    )
    end: float = Field(
        ge=0.0,
        description="End timestamp of the segment in seconds.",
    )
    words: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of word-level metadata dictionaries (word, start, end, probability).",
    )
    sentiment_score: float = Field(
        default=0.0,
        description="Assigned sentiment intensity score (default: 0.0).",
    )


class HypeScore(BaseModel):
    """Calculated engagement/hype score and composite components at a specific timestamp."""

    timestamp: float = Field(
        ge=0.0,
        description="Timestamp in seconds.",
    )
    score: float = Field(
        description="Final weighted composite hype score.",
    )
    audio_component: float = Field(
        description="Normalized audio excitation component score.",
    )
    motion_component: float = Field(
        description="Normalized visual motion dynamic score.",
    )
    face_component: float = Field(
        description="Normalized facial engagement score.",
    )
    sentiment_component: float = Field(
        description="Normalized speech sentiment score.",
    )


class ClipWindow(BaseModel):
    """Candidate temporal slice selected for clip generation."""

    start_time: float = Field(
        ge=0.0,
        description="Start timestamp of the clip window in seconds.",
    )
    end_time: float = Field(
        ge=0.0,
        description="End timestamp of the clip window in seconds.",
    )
    duration: float = Field(
        gt=0.0,
        description="Duration of the clip window in seconds.",
    )
    hype_score: float = Field(
        description="Aggregate hype score for this candidate window.",
    )
    peak_timestamp: float = Field(
        ge=0.0,
        description="Timestamp of the maximum hype peak within the candidate window.",
    )


class ClipResult(BaseModel):
    """Information and artifact file paths for a successfully rendered clip."""

    clip_path: str = Field(
        description="Filesystem path to the rendered output clip video file.",
    )
    start_time: float = Field(
        ge=0.0,
        description="Source video start timestamp in seconds.",
    )
    end_time: float = Field(
        ge=0.0,
        description="Source video end timestamp in seconds.",
    )
    duration: float = Field(
        gt=0.0,
        description="Duration of the rendered clip in seconds.",
    )
    hype_score: float = Field(
        description="Overall hype score assigned to this clip.",
    )
    thumbnail_path: str | None = Field(
        default=None,
        description="Filesystem path to generated thumbnail image, if created.",
    )
    subtitle_path: str | None = Field(
        default=None,
        description="Filesystem path to generated subtitle file, if created.",
    )


class PipelineResult(BaseModel):
    """Final execution summary of the complete Clizzer video processing pipeline."""

    clips: list[ClipResult] = Field(
        default_factory=list,
        description="List of produced clip results.",
    )
    total_duration: float = Field(
        ge=0.0,
        description="Total duration of the input source video in seconds.",
    )
    processing_time: float = Field(
        ge=0.0,
        description="Total wall-clock processing time in seconds.",
    )
    video_title: str = Field(
        default="",
        description="Title or identifier of the processed video.",
    )
