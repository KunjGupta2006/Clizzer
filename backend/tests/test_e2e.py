import json
from pathlib import Path
import re
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.models import JobParams
from backend.app.ingest.ffmpeg_utils import (
    generate_synthetic_video,
    validate_source,
    get_duration,
    run_ffmpeg,
)
from backend.app.core.orchestrator import execute_pipeline
from backend.app.config import settings


@pytest.fixture(scope="session")
def e2e_synthetic_video(tmp_path_factory) -> Path:
    temp_dir = tmp_path_factory.mktemp("e2e_fixture")
    src = temp_dir / "e2e_source_25s.mp4"
    return generate_synthetic_video(src, duration=25.0, with_audio=True)


def test_e2e_pipeline_and_api(e2e_synthetic_video: Path):
    client = TestClient(app)

    # 1. Health check (no API keys required)
    resp = client.get("/health")
    assert resp.status_code == 200
    health_data = resp.json()
    assert health_data["status"] == "healthy"

    # 2. Permission check: permission_confirmed=False returns HTTP 400
    bad_resp = client.post(
        "/jobs",
        json={
            "source_url": "https://youtube.com/watch?v=mock",
            "permission_confirmed": False,
        },
    )
    assert bad_resp.status_code == 400

    # 3. Create valid job with synthetic video file
    job_payload = {
        "file_path": str(e2e_synthetic_video),
        "num_clips": 1,
        "min_len": 10,
        "max_len": 20,
        "aspect": "9:16",
        "mode": "auto",
        "captions": True,
        "permission_confirmed": True,
    }
    create_resp = client.post("/jobs", json=job_payload)
    assert create_resp.status_code == 201
    job_id = create_resp.json()["job_id"]
    assert job_id.startswith("job_")

    # 4. Run pipeline synchronously for test verification
    params = JobParams(**job_payload)
    rendered_clips = execute_pipeline(job_id, params)

    # 5. Verify job is marked 'done'
    job_status_resp = client.get(f"/jobs/{job_id}")
    assert job_status_resp.status_code == 200
    job_data = job_status_resp.json()
    assert job_data["status"] == "done"
    assert job_data["progress"] == 1.0

    # 6. Verify at least 1 clip is generated and listed
    clips_resp = client.get(f"/jobs/{job_id}/clips")
    assert clips_resp.status_code == 200
    clips_list = clips_resp.json()
    assert len(clips_list) >= 1

    first_clip_meta = clips_list[0]
    clip_file = Path(first_clip_meta["file"])
    assert clip_file.exists()
    assert clip_file.stat().st_size > 0

    # 7. G2 requirement: 1080x1920 resolution
    info = validate_source(clip_file)
    v_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert v_stream["width"] == 1080
    assert v_stream["height"] == 1920

    # 8. G2 requirement: loudness -14 +/- 1.5 LUFS
    loudnorm_cmd = [
        "ffmpeg",
        "-i", str(clip_file),
        "-af", "loudnorm=I=-14:LRA=11:TP=-1.5:print_format=json",
        "-f", "null",
        "-",
    ]
    proc = run_ffmpeg(loudnorm_cmd, timeout=30.0)
    json_match = re.search(r"\{\s*\"input_i\"\s*:\s*\"([^\"]+)\"", proc.stderr)
    assert json_match is not None, "Failed to parse loudnorm JSON from ffmpeg output"
    measured_lufs = float(json_match.group(1))
    assert abs(measured_lufs - (-14.0)) <= 1.5, f"Loudness {measured_lufs} not within -14 +/- 1.5 LUFS"

    # 9. Verify curve endpoint returns valid data
    curve_resp = client.get(f"/jobs/{job_id}/curve")
    assert curve_resp.status_code == 200
    curve_data = curve_resp.json()
    assert curve_data["length"] > 0
    assert "fused" in curve_data
    assert "audio" in curve_data
