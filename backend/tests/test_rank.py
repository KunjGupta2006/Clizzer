import numpy as np
import pytest
from backend.app.models import Candidate, SeriesBundle, Word
from backend.app.rank.fusion import fuse_signals
from backend.app.rank.dedup import deduplicate_candidates, temporal_iou
from backend.app.rank.select import select_top_clips


def test_planted_peaks_recovered_within_2s():
    duration = 150.0
    length = int(duration)

    audio = np.zeros(length, dtype=np.float32)
    chat = np.zeros(length, dtype=np.float32)
    text = np.zeros(length, dtype=np.float32)
    visual = np.zeros(length, dtype=np.float32)
    face = np.zeros(length, dtype=np.float32)

    # Plant peak 1 at t=35.0s
    peak1_t = 35.0
    audio[34:37] = 2.8
    chat[34:37] = 2.5

    # Plant peak 2 at t=100.0s
    peak2_t = 100.0
    audio[99:102] = 2.9
    visual[99:102] = 2.7

    bundle = SeriesBundle.from_numpy(
        duration=duration,
        audio=audio,
        chat=chat,
        text=text,
        visual=visual,
        face=face,
    )

    words = [
        Word(start=30.0, end=34.0, text="Watch this insane trick!"),
        Word(start=35.0, end=40.0, text="Oh my god he did it!"),
        Word(start=95.0, end=99.0, text="Here comes the car jump!"),
        Word(start=100.0, end=105.0, text="Huge explosion!"),
    ]

    clips, shortfall = select_top_clips(
        bundle=bundle,
        words=words,
        num_clips=2,
        min_len=20.0,
        max_len=40.0,
        mode="auto",
        quality_floor=0.8,
    )

    assert len(clips) == 2
    recovered_peaks = [c.peak_t for c in clips]

    # Check recovery within +/- 2 seconds
    assert any(abs(p - peak1_t) <= 2.0 for p in recovered_peaks), f"Peak 1 near {peak1_t} not found in {recovered_peaks}"
    assert any(abs(p - peak2_t) <= 2.0 for p in recovered_peaks), f"Peak 2 near {peak2_t} not found in {recovered_peaks}"


def test_deduplication():
    # Candidates with high temporal IoU > 0.3
    c1 = Candidate(start=10.0, end=40.0, peak_t=25.0, fused_score=2.5)
    c2 = Candidate(start=12.0, end=42.0, peak_t=27.0, fused_score=1.8)  # Highly overlapping
    c3 = Candidate(start=80.0, end=110.0, peak_t=95.0, fused_score=2.2) # Far away

    assert temporal_iou(c1, c2) > 0.5
    deduped = deduplicate_candidates([c1, c2, c3], min_spacing_sec=45.0, iou_threshold=0.3)

    assert len(deduped) == 2
    assert c1 in deduped
    assert c2 not in deduped
    assert c3 in deduped


def test_quality_floor_and_shortfall():
    duration = 100.0
    length = int(duration)

    # Flat, low signal
    zeros = np.zeros(length, dtype=np.float32)
    bundle = SeriesBundle.from_numpy(
        duration=duration,
        audio=zeros,
        chat=zeros,
        text=zeros,
        visual=zeros,
        face=zeros,
    )

    clips, shortfall = select_top_clips(
        bundle=bundle,
        words=[],
        num_clips=5,
        quality_floor=0.8,
    )

    # Should return fewer than requested (0 in this case) and not pad
    assert len(clips) < 5
    assert shortfall is not None
    assert "quality floor" in shortfall


def test_determinism_seed_42():
    duration = 100.0
    length = int(duration)
    np.random.seed(123)
    audio = np.random.randn(length).astype(np.float32)
    chat = np.random.randn(length).astype(np.float32)

    bundle1 = SeriesBundle.from_numpy(duration=duration, audio=audio, chat=chat, text=np.zeros(length), visual=np.zeros(length), face=np.zeros(length))
    bundle2 = SeriesBundle.from_numpy(duration=duration, audio=audio, chat=chat, text=np.zeros(length), visual=np.zeros(length), face=np.zeros(length))

    clips1, _ = select_top_clips(bundle1, words=[], num_clips=3, seed=42, quality_floor=-10.0)
    clips2, _ = select_top_clips(bundle2, words=[], num_clips=3, seed=42, quality_floor=-10.0)

    assert len(clips1) == len(clips2)
    for c1, c2 in zip(clips1, clips2):
        assert c1.peak_t == pytest.approx(c2.peak_t, abs=1e-4)
        assert c1.fused_score == pytest.approx(c2.fused_score, abs=1e-4)
        assert c1.start == pytest.approx(c2.start, abs=1e-4)
        assert c1.end == pytest.approx(c2.end, abs=1e-4)


def test_modality_renormalization():
    bundle = SeriesBundle.from_numpy(
        duration=30.0,
        audio=np.ones(30),
        chat=np.zeros(30),
        text=np.zeros(30),
        visual=np.ones(30),
        face=np.zeros(30),
        available={"audio": True, "visual": True, "chat": False, "text": False, "face": False},
    )
    fused, weights = fuse_signals(bundle, mode="auto")
    # Only audio and visual available
    assert weights["chat"] == 0.0
    assert weights["text"] == 0.0
    assert weights["face"] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-5)
    assert weights["audio"] > 0.0
    assert weights["visual"] > 0.0
