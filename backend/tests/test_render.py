import json
from pathlib import Path
import re
import pytest
from backend.app.models import Clip, Word
from backend.app.ingest.ffmpeg_utils import (
    generate_synthetic_video,
    validate_source,
    get_duration,
    run_ffmpeg,
)
from backend.app.render.render import render_clip


@pytest.fixture(scope="session")
def render_source_video(tmp_path_factory) -> Path:
    temp_dir = tmp_path_factory.mktemp("render_fixture")
    src = temp_dir / "source_25s.mp4"
    # Generate 25-second synthetic video with 1kHz tone
    return generate_synthetic_video(src, duration=25.0, with_audio=True)


def test_render_clip_output(render_source_video: Path, tmp_path: Path):
    job_id = "test_render_job"
    clip = Clip(
        start=5.0,
        end=15.0,  # 10 second clip
        peak_t=10.0,
        fused_score=2.3,
        title="Epic GTA 6 Moment",
        hook="Wait for the jump!",
        hashtags=["#GTA6", "#Gaming"],
    )
    words = [
        Word(start=6.0, end=7.5, text="Watch"),
        Word(start=7.6, end=8.5, text="this"),
        Word(start=8.6, end=10.0, text="jump"),
        Word(start=10.1, end=12.0, text="right now!"),
    ]

    rendered_clip = render_clip(
        job_id=job_id,
        clip_idx=1,
        source_path=render_source_video,
        clip=clip,
        words=words,
        aspect="9:16",
        out_dir=tmp_path / "out",
        attribution="@RockstarGames",
        captions=True,
    )

    clip_file = Path(rendered_clip.file)
    assert clip_file.exists()
    assert clip_file.stat().st_size > 0

    # 1. Verify resolution is exactly 1080x1920
    info = validate_source(clip_file)
    v_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert v_stream["width"] == 1080
    assert v_stream["height"] == 1920

    # 2. Verify duration is within +/- 0.5 s of target (10.0s)
    dur = get_duration(clip_file)
    expected_dur = clip.end - clip.start
    assert abs(dur - expected_dur) <= 0.5

    # 3. Verify loudness is ~ -14 LUFS (+/- 1.5 LUFS)
    # Run ffmpeg loudnorm measurement
    cmd = [
        "ffmpeg",
        "-i", str(clip_file),
        "-af", "loudnorm=I=-14:LRA=11:TP=-1.5:print_format=json",
        "-f", "null",
        "-",
    ]
    proc = run_ffmpeg(cmd, timeout=30.0)
    stderr = proc.stderr

    # Parse measured input_i
    json_match = re.search(r"\{\s*\"input_i\"\s*:\s*\"([^\"]+)\"", stderr)
    if json_match:
        measured_lufs = float(json_match.group(1))
        # Spec requirement: loudness -14 +/- 1.5 LUFS
        assert abs(measured_lufs - (-14.0)) <= 1.5, f"Measured LUFS {measured_lufs} not within -14 +/- 1.5"

    # 4. Verify auxiliary files created
    assert Path(rendered_clip.srt).exists()
    assert Path(rendered_clip.thumb).exists()
    json_meta = clip_file.with_suffix(".json")
    assert json_meta.exists()
    meta_data = json.loads(json_meta.read_text(encoding="utf-8"))
    assert meta_data["title"] == "Epic GTA 6 Moment"
    assert meta_data["attribution"] == "@RockstarGames"
