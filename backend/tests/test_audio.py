import math
from pathlib import Path
import numpy as np
import pytest
from backend.app.models import Word
from backend.app.ingest.ffmpeg_utils import generate_synthetic_video, extract_audio
from backend.app.audio.features import compute_audio_features, rolling_median_mad
from backend.app.audio.diarize import diarize_audio, assign_speakers_to_words
from backend.app.audio.events import detect_audio_events
from backend.app.audio.asr import transcribe_audio


@pytest.fixture(scope="session")
def sample_wav(tmp_path_factory) -> Path:
    temp_dir = tmp_path_factory.mktemp("audio_test")
    vid = temp_dir / "temp_vid.mp4"
    wav = temp_dir / "audio_15s.wav"
    generate_synthetic_video(vid, duration=15.0, with_audio=True)
    extract_audio(vid, wav)
    return wav


def test_rolling_median_mad():
    arr = np.array([1.0, 2.0, 10.0, 2.0, 1.0, 1.0], dtype=np.float32)
    norm = rolling_median_mad(arr, window_size=5)
    assert len(norm) == len(arr)
    assert not np.isnan(norm).any()
    assert not np.isinf(norm).any()
    assert (norm >= -3.0).all() and (norm <= 3.0).all()
    # Spike at index 2 should have high positive z-score
    assert norm[2] > norm[0]


def test_compute_audio_features(sample_wav: Path):
    duration = 15.0
    words = [
        Word(start=1.0, end=1.5, text="Welcome"),
        Word(start=1.6, end=2.0, text="to"),
        Word(start=2.1, end=2.8, text="Vice"),
        Word(start=2.9, end=3.5, text="City"),
    ]
    raw_feats, audio_score, music_ratio, available = compute_audio_features(
        sample_wav,
        duration=duration,
        words=words,
    )

    expected_len = int(math.ceil(duration))
    assert len(audio_score) == expected_len
    assert len(music_ratio) == expected_len
    assert not np.isnan(audio_score).any()
    assert not np.isinf(audio_score).any()
    assert not np.isnan(music_ratio).any()
    assert (audio_score >= -3.0).all() and (audio_score <= 3.0).all()
    assert (music_ratio >= 0.0).all() and (music_ratio <= 1.0).all()
    assert available is True


def test_diarization_fallback_hf_token_unset(sample_wav: Path):
    # Tested fallback: HF_TOKEN unset -> returns single_speaker_fallback
    intervals, method = diarize_audio(sample_wav, hf_token=None)
    assert len(intervals) >= 1
    assert method == "single_speaker_fallback"
    assert intervals[0][2] == "SPEAKER_00"


def test_assign_speakers():
    words = [
        Word(start=0.5, end=1.0, text="Hey"),
        Word(start=5.0, end=6.0, text="There"),
    ]
    intervals = [
        (0.0, 2.0, "SPEAKER_01"),
        (4.0, 8.0, "SPEAKER_02"),
    ]
    assigned = assign_speakers_to_words(words, intervals)
    assert assigned[0].speaker == "SPEAKER_01"
    assert assigned[1].speaker == "SPEAKER_02"


def test_detect_audio_events(sample_wav: Path):
    duration = 10.0
    excitement, music_ratio, available = detect_audio_events(sample_wav, duration)
    assert len(excitement) == 10
    assert len(music_ratio) == 10
    assert not np.isnan(excitement).any()
    assert not np.isnan(music_ratio).any()


def test_asr_graceful_execution(sample_wav: Path):
    from unittest.mock import patch, MagicMock

    # 1. Fallback when model is None
    with patch("backend.app.audio.asr.get_whisper_model", return_value=None):
        segments, words = transcribe_audio(sample_wav)
        assert segments == []
        assert words == []

    # 2. Mocked primary path
    mock_model = MagicMock()
    mock_seg = MagicMock()
    mock_seg.start = 0.5
    mock_seg.end = 2.0
    mock_seg.text = "Mock speech"
    mock_word = MagicMock()
    mock_word.start = 0.5
    mock_word.end = 1.0
    mock_word.word = "Mock"
    mock_word.probability = 0.95
    mock_seg.words = [mock_word]

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_model.transcribe.return_value = ([mock_seg], mock_info)

    with patch("backend.app.audio.asr.get_whisper_model", return_value=mock_model):
        segments, words = transcribe_audio(sample_wav)
        assert len(segments) == 1
        assert segments[0].text == "Mock speech"
        assert len(words) == 1
        assert words[0].text == "Mock"
