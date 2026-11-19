import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from backend.app.errors import SourceUnavailable
from backend.app.ingest.ffmpeg_utils import (
    generate_synthetic_video,
    validate_source,
    get_duration,
    extract_audio,
    create_proxy_video,
)
from backend.app.ingest.downloader import download_video
from backend.app.ingest.chat_fetch import fetch_chat
from backend.app.ingest.chunker import plan_chunks


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> Path:
    temp_dir = tmp_path_factory.mktemp("synthetic")
    video_path = temp_dir / "test_source.mp4"
    return generate_synthetic_video(video_path, duration=10.0, with_audio=True)


def test_validate_and_duration(synthetic_video: Path):
    info = validate_source(synthetic_video)
    assert "streams" in info
    assert len(info["streams"]) >= 2  # video + audio

    duration = get_duration(synthetic_video)
    assert 9.0 <= duration <= 11.0


def test_validate_corrupted(tmp_path: Path):
    corrupt_file = tmp_path / "corrupt.mp4"
    corrupt_file.write_text("not a video")
    with pytest.raises(SourceUnavailable):
        validate_source(corrupt_file)


def test_extract_audio(synthetic_video: Path, tmp_path: Path):
    out_wav = tmp_path / "audio.wav"
    extracted = extract_audio(synthetic_video, out_wav)
    assert extracted.exists()
    assert extracted.stat().st_size > 0

    info = validate_source(extracted)
    audio_stream = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert audio_stream["sample_rate"] == "16000"
    assert audio_stream["channels"] == 1


def test_create_proxy_video(synthetic_video: Path, tmp_path: Path):
    out_proxy = tmp_path / "proxy.mp4"
    proxy = create_proxy_video(synthetic_video, out_proxy)
    assert proxy.exists()
    assert proxy.stat().st_size > 0

    info = validate_source(proxy)
    video_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert video_stream["height"] == 360


def test_downloader_local_file(synthetic_video: Path, tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    source_mp4, meta = download_video(
        job_id="job_local_1",
        source_url=None,
        file_path=str(synthetic_video),
        jobs_dir=jobs_dir,
    )
    assert source_mp4.exists()
    assert meta["duration"] > 0
    assert meta["job_id"] == "job_local_1"


def test_downloader_mocked_ytdlp(synthetic_video: Path, tmp_path: Path):
    jobs_dir = tmp_path / "jobs"

    with patch("backend.app.ingest.downloader.yt_dlp") as mock_yt:
        mock_instance = MagicMock()
        mock_yt.YoutubeDL.return_value.__enter__.return_value = mock_instance

        # Simulate downloaded file by copying fixture to expected destination
        def mock_extract(url, download=True):
            dest = jobs_dir / "job_yt_1" / "source.mp4"
            dest.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(synthetic_video, dest)
            return {"title": "Test YouTube Video", "duration": 10.0, "uploader": "TestUser"}

        mock_instance.extract_info.side_effect = mock_extract

        source_mp4, meta = download_video(
            job_id="job_yt_1",
            source_url="https://youtube.com/watch?v=mock123",
            file_path=None,
            jobs_dir=jobs_dir,
        )
        assert source_mp4.exists()
        assert meta["title"] == "Test YouTube Video"
        assert meta["duration"] == 10.0


def test_downloader_missing_file(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    with pytest.raises(SourceUnavailable):
        download_video("job_fail", None, str(tmp_path / "nonexistent.mp4"), jobs_dir)


def test_chat_fetch_fallback(tmp_path: Path):
    out_json = tmp_path / "chat.json"
    msgs = fetch_chat(None, out_json)
    assert msgs == []

    # Unreachable URL returns empty list without raising
    msgs = fetch_chat("https://invalid-url.fake/video", out_json)
    assert isinstance(msgs, list)


def test_chunker_planning():
    # <= 2h (7200s) -> single chunk
    chunks = plan_chunks(3600.0)
    assert chunks == [(0.0, 3600.0)]

    # 3 hours (10800s) -> multi chunk with overlap
    chunks_3h = plan_chunks(10800.0, window_sec=1800.0, overlap_sec=30.0)
    assert len(chunks_3h) > 1
    assert chunks_3h[0] == (0.0, 1800.0)
    assert chunks_3h[1][0] == 1770.0  # 1800 - 30
