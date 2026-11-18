"""Unit and integration tests for Clizzer core modules (ClipGenerator and SubtitleGenerator)."""

import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import pysubs2
import pytest

from backend.core.clip_generator import ClipGenerator
from backend.core.subtitle_gen import SubtitleGenerator


@pytest.fixture
def sample_words():
    return [
        {"word": "Welcome", "start": 0.1, "end": 0.5},
        {"word": "to", "start": 0.5, "end": 0.7},
        {"word": "Clizzer", "start": 0.7, "end": 1.2},
        {"word": "video", "start": 1.2, "end": 1.6},
        {"word": "highlights", "start": 1.6, "end": 2.2},
        {"word": "generator", "start": 2.2, "end": 2.8},
        {"word": "automatic", "start": 3.0, "end": 3.5},
        {"word": "clips", "start": 3.5, "end": 4.0},
    ]


@pytest.fixture
def synthetic_video(tmp_path):
    video_path = tmp_path / "test_synth.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=size=640x360:rate=25:duration=4",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=4",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        str(video_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return video_path


def test_subtitle_generator_init():
    sg = SubtitleGenerator()
    assert sg.font_name == "Big Shoulders Display"
    assert sg.outline_width == 3.0
    assert sg.shadow_width == 2.0
    assert sg.alignment == pysubs2.Alignment.BOTTOM_CENTER
    assert sg.white_color == pysubs2.Color(255, 255, 255)
    assert sg.gta_yellow == pysubs2.Color(245, 184, 0)


def test_subtitle_group_words(sample_words):
    sg = SubtitleGenerator()
    lines = sg._group_words_into_lines(sample_words, max_per_line=6)
    assert len(lines) == 2
    assert len(lines[0]) == 6
    assert len(lines[1]) == 2
    assert lines[0][0]["word"] == "Welcome"
    assert lines[1][0]["word"] == "automatic"


def test_subtitle_karaoke_generation(sample_words, tmp_path):
    sg = SubtitleGenerator()
    out_file = tmp_path / "karaoke.ass"
    ret = sg.generate_subtitles(sample_words, out_file, style="karaoke")
    assert Path(ret).exists()

    subs = pysubs2.load(ret)
    assert len(subs.events) == 2
    assert r"\k" in subs.events[0].text
    assert "karaoke" in subs.styles


def test_subtitle_gta_generation(sample_words, tmp_path):
    sg = SubtitleGenerator()
    out_file = tmp_path / "gta.ass"
    ret = sg.generate_subtitles(sample_words, out_file, style="gta")
    assert Path(ret).exists()

    subs = pysubs2.load(ret)
    assert len(subs.events) == 2
    assert subs.events[0].text == "WELCOME TO CLIZZER VIDEO HIGHLIGHTS GENERATOR"
    assert "gta" in subs.styles
    assert subs.styles["gta"].bold == True
    assert subs.styles["gta"].primarycolor == pysubs2.Color(245, 184, 0)


def test_clip_generator_cut_basic(synthetic_video, tmp_path):
    out_dir = tmp_path / "clips"
    cg = ClipGenerator(output_dir=out_dir)
    dest = out_dir / "basic.mp4"
    ret = cg._cut_clip(synthetic_video, 1.0, 3.0, dest, reframe=False)
    assert Path(ret).exists()

    cap = cv2.VideoCapture(str(dest))
    assert cap.isOpened()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert w == 640 and h == 360


def test_clip_generator_cut_reframe(synthetic_video, tmp_path):
    out_dir = tmp_path / "clips"
    cg = ClipGenerator(output_dir=out_dir)
    dest = out_dir / "reframe.mp4"
    ret = cg._cut_clip(synthetic_video, 1.0, 3.0, dest, reframe=True, face_center=(0.4, 0.5))
    assert Path(ret).exists()

    cap = cv2.VideoCapture(str(dest))
    assert cap.isOpened()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert h == 360
    assert abs((w / h) - (9 / 16)) < 0.05


def test_clip_generator_thumbnail(synthetic_video, tmp_path):
    out_dir = tmp_path / "clips"
    cg = ClipGenerator(output_dir=out_dir)
    thumb_path = out_dir / "thumb.jpg"
    ret = cg._generate_thumbnail(synthetic_video, 1.5, thumb_path)
    assert Path(ret).exists()
    img = cv2.imread(str(ret))
    assert img is not None


def test_clip_generator_burn_subtitles(synthetic_video, sample_words, tmp_path):
    out_dir = tmp_path / "clips"
    cg = ClipGenerator(output_dir=out_dir)
    sg = SubtitleGenerator()
    ass_path = out_dir / "subs.ass"
    sg.generate_subtitles(sample_words, ass_path, style="karaoke")

    burned = out_dir / "burned.mp4"
    ret = cg._burn_subtitles(synthetic_video, ass_path, burned)
    assert Path(ret).exists()


def test_clip_generator_full_pipeline(synthetic_video, sample_words, tmp_path):
    out_dir = tmp_path / "clips"
    cg = ClipGenerator(output_dir=out_dir)
    sg = SubtitleGenerator()
    ass_path = out_dir / "clip_subs.ass"
    sg.generate_subtitles(sample_words, ass_path, style="karaoke")

    windows = [
        {"start": 0.5, "end": 2.0, "hype_score": 0.95},
        {"start": 1.5, "end": 3.5, "hype_score": 0.88},
    ]
    face_track = [
        {"timestamp": 1.0, "center_x": 0.5, "center_y": 0.5},
        {"timestamp": 2.5, "center_x": 0.45, "center_y": 0.5},
    ]

    progress_log = []
    def on_progress(p, msg=""):
        progress_log.append((p, msg))

    clips = cg.generate_clips(
        video_path=synthetic_video,
        clip_windows=windows,
        face_track=face_track,
        subtitle_paths=[ass_path, ass_path],
        progress_callback=on_progress,
        reframe=True,
    )

    assert len(clips) == 2
    for c in clips:
        assert Path(c["clip_path"]).exists()
        assert Path(c["thumbnail_path"]).exists()
        assert "start" in c and "end" in c and "duration" in c and "hype_score" in c

    assert len(progress_log) >= 2
