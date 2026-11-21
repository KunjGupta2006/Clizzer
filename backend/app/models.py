from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlmodel import Field as SQLField, SQLModel, Column, JSON


class JobParams(BaseModel):
    source_url: Optional[str] = None
    file_path: Optional[str] = None
    num_clips: int = Field(default=5, ge=1, le=30)
    min_len: int = Field(default=20, ge=5, le=120)
    max_len: int = Field(default=60, ge=10, le=300)
    aspect: str = Field(default="9:16")  # "9:16", "1:1", "16:9"
    layout: str = Field(default="single")  # "single", "game_top", "cam_top", "pip"
    regions: Optional[List[Dict[str, Any]]] = None  # e.g. [{"label": "cam", "x": 0.7, "y": 0.05, "w": 0.25, "h": 0.3}, ...]
    mode: str = Field(default="auto")    # "auto", "hype", "funny", "informational"
    keywords: List[str] = Field(default_factory=list)
    captions: bool = True
    permission_confirmed: bool = False

    @model_validator(mode="after")
    def validate_inputs(self) -> "JobParams":
        if not self.permission_confirmed:
            raise ValueError("permission_confirmed must be true to process content")
        if not self.source_url and not self.file_path:
            raise ValueError("Either source_url or file_path must be provided")
        if self.min_len > self.max_len:
            raise ValueError("min_len cannot be greater than max_len")
        return self


class Word(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    conf: float = 1.0


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    lang: str = "en"


class ChatMsg(BaseModel):
    t: float
    author: str
    text: str
    emotes: List[str] = Field(default_factory=list)


class FaceFrame(BaseModel):
    t: float
    bbox: List[float]  # [x, y, w, h] normalized or pixel coords
    mouth_open: float = 0.0
    expr_score: float = 0.0


class FaceTrack(BaseModel):
    person_id: int
    frames: List[FaceFrame] = Field(default_factory=list)


class Candidate(BaseModel):
    start: float
    end: float
    peak_t: float
    fused_score: float
    breakdown: Dict[str, float] = Field(default_factory=dict)  # audio, chat, text, visual, face
    reason: str = ""
    music_ratio: float = 0.0


class Clip(Candidate):
    title: str = Field(..., max_length=70)
    hook: str = ""
    hashtags: List[str] = Field(default_factory=list)
    file: str = ""
    srt: str = ""
    thumb: str = ""
    source_url: str = ""
    attribution: str = ""

    @field_validator("title", mode="before")
    @classmethod
    def clamp_title(cls, v: Any) -> str:
        s = str(v)
        if len(s) > 70:
            return s[:67] + "..."
        return s


class SeriesBundle(BaseModel):
    duration: float
    length: int
    audio: List[float] = Field(default_factory=list)
    chat: List[float] = Field(default_factory=list)
    text: List[float] = Field(default_factory=list)
    visual: List[float] = Field(default_factory=list)
    face: List[float] = Field(default_factory=list)
    fused: List[float] = Field(default_factory=list)
    raw_features: Dict[str, List[float]] = Field(default_factory=dict)
    available: Dict[str, bool] = Field(
        default_factory=lambda: {
            "audio": True,
            "chat": True,
            "text": True,
            "visual": True,
            "face": True,
        }
    )
    fallback_reasons: Dict[str, Optional[str]] = Field(default_factory=dict)

    @classmethod
    def from_numpy(
        cls,
        duration: float,
        audio: np.ndarray,
        chat: np.ndarray,
        text: np.ndarray,
        visual: np.ndarray,
        face: np.ndarray,
        available: Optional[Dict[str, bool]] = None,
        raw_features: Optional[Dict[str, np.ndarray]] = None,
        fallback_reasons: Optional[Dict[str, Optional[str]]] = None,
    ) -> "SeriesBundle":
        length = int(np.ceil(duration))
        pad = lambda arr: np.pad(arr, (0, max(0, length - len(arr))))[:length]
        raw_dict = {}
        if raw_features:
            for k, v in raw_features.items():
                raw_dict[k] = [float(x) for x in pad(v)]

        return cls(
            duration=float(duration),
            length=length,
            audio=[float(x) for x in pad(audio)],
            chat=[float(x) for x in pad(chat)],
            text=[float(x) for x in pad(text)],
            visual=[float(x) for x in pad(visual)],
            face=[float(x) for x in pad(face)],
            fused=[0.0] * length,
            raw_features=raw_dict,
            available=available or {"audio": True, "chat": True, "text": True, "visual": True, "face": True},
            fallback_reasons=fallback_reasons or {},
        )

    def to_arrays(self) -> Dict[str, np.ndarray]:
        return {
            "audio": np.array(self.audio, dtype=np.float32),
            "chat": np.array(self.chat, dtype=np.float32),
            "text": np.array(self.text, dtype=np.float32),
            "visual": np.array(self.visual, dtype=np.float32),
            "face": np.array(self.face, dtype=np.float32),
            "fused": np.array(self.fused, dtype=np.float32),
        }


class Job(SQLModel, table=True):
    id: str = SQLField(default=None, primary_key=True)
    status: str = SQLField(default="queued")  # queued, downloading, transcribing, analyzing, ranking, rendering, done, failed
    progress: float = SQLField(default=0.0)
    stage: str = SQLField(default="queued")
    timings_json: str = SQLField(default="{}")
    params_json: str = SQLField(default="{}")
    error: Optional[str] = SQLField(default=None)
    created_at: str = SQLField(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def timings(self) -> Dict[str, Any]:
        try:
            return json.loads(self.timings_json)
        except Exception:
            return {}

    @timings.setter
    def timings(self, value: Dict[str, Any]) -> None:
        self.timings_json = json.dumps(value)

    @property
    def params(self) -> Dict[str, Any]:
        try:
            return json.loads(self.params_json)
        except Exception:
            return {}

    @params.setter
    def params(self, value: Dict[str, Any]) -> None:
        self.params_json = json.dumps(value)
