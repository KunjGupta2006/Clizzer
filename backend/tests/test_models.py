import pytest
import numpy as np
from backend.app.models import (
    JobParams,
    Word,
    Segment,
    ChatMsg,
    FaceFrame,
    FaceTrack,
    Candidate,
    Clip,
    SeriesBundle,
    Job,
)
from backend.app.errors import (
    ClizzerError,
    SourceUnavailable,
    NoSpeech,
    ChatUnavailable,
    RenderFailed,
    LLMUnavailable,
)


def test_job_params_valid():
    params = JobParams(
        source_url="https://youtube.com/watch?v=dQw4w9WgXcQ",
        num_clips=5,
        min_len=20,
        max_len=60,
        aspect="9:16",
        mode="hype",
        permission_confirmed=True,
    )
    assert params.num_clips == 5
    assert params.permission_confirmed is True


def test_job_params_requires_permission():
    with pytest.raises(ValueError, match="permission_confirmed"):
        JobParams(
            source_url="https://youtube.com/watch?v=dQw4w9WgXcQ",
            permission_confirmed=False,
        )


def test_job_params_requires_source():
    with pytest.raises(ValueError, match="Either source_url or file_path"):
        JobParams(
            permission_confirmed=True,
        )


def test_job_params_min_max_len():
    with pytest.raises(ValueError, match="min_len cannot be greater"):
        JobParams(
            source_url="https://youtube.com",
            min_len=70,
            max_len=30,
            permission_confirmed=True,
        )


def test_word_and_segment():
    w = Word(start=1.0, end=1.5, text="Hello", speaker="SPEAKER_00", conf=0.98)
    assert w.text == "Hello"
    assert w.conf == 0.98

    s = Segment(start=0.0, end=5.0, text="Hello world", speaker="SPEAKER_00", lang="en")
    assert s.lang == "en"
    assert s.start == 0.0


def test_chat_msg():
    msg = ChatMsg(t=12.5, author="gamer123", text="POGGERS clip that!")
    assert msg.t == 12.5
    assert msg.emotes == []


def test_facetrack():
    f1 = FaceFrame(t=0.5, bbox=[10.0, 20.0, 100.0, 100.0], mouth_open=0.4, expr_score=0.8)
    track = FaceTrack(person_id=1, frames=[f1])
    assert track.person_id == 1
    assert len(track.frames) == 1
    assert track.frames[0].mouth_open == 0.4


def test_candidate_and_clip():
    cand = Candidate(
        start=10.0,
        end=40.0,
        peak_t=25.0,
        fused_score=2.45,
        breakdown={"audio": 1.2, "chat": 2.0, "text": 0.5, "visual": 0.8, "face": 0.1},
        reason="Loud scream and high chat velocity",
        music_ratio=0.1,
    )
    assert cand.fused_score == 2.45

    long_title = "A" * 80
    clip = Clip(
        start=10.0,
        end=40.0,
        peak_t=25.0,
        fused_score=2.45,
        title=long_title,
        hook="Wait for it...",
        hashtags=["#gaming", "#epic", "#gta6"],
        file="clip_01.mp4",
        srt="clip_01.srt",
        thumb="clip_01.jpg",
    )
    assert len(clip.title) <= 70
    assert clip.title.endswith("...")


def test_series_bundle_from_numpy():
    duration = 5.2  # ceil -> 6
    audio = np.array([0.1, 0.5, 0.9, 0.2, 0.0])
    chat = np.array([0.0, 1.0, 2.0])
    text = np.zeros(6)
    visual = np.ones(6)
    face = np.zeros(6)

    bundle = SeriesBundle.from_numpy(
        duration=duration,
        audio=audio,
        chat=chat,
        text=text,
        visual=visual,
        face=face,
        available={"audio": True, "chat": True, "text": True, "visual": True, "face": False},
        fallback_reasons={"face": "no faces detected"},
    )
    assert bundle.length == 6
    assert len(bundle.audio) == 6
    assert bundle.audio[5] == 0.0  # padded
    assert bundle.available["face"] is False
    assert bundle.fallback_reasons["face"] == "no faces detected"

    arrs = bundle.to_arrays()
    assert isinstance(arrs["audio"], np.ndarray)
    assert len(arrs["audio"]) == 6


def test_job_sqlmodel():
    job = Job(id="job_test_123", status="queued", progress=0.0)
    assert job.timings == {}
    job.timings = {"ingest": 1.25}
    assert job.timings["ingest"] == 1.25

    job.params = {"num_clips": 3}
    assert job.params["num_clips"] == 3


def test_custom_errors():
    errors = [
        SourceUnavailable("Source unavailable"),
        NoSpeech("No speech"),
        ChatUnavailable("No chat"),
        RenderFailed("Render failed"),
        LLMUnavailable("LLM down"),
    ]
    for err in errors:
        assert isinstance(err, ClizzerError)
        assert err.message
