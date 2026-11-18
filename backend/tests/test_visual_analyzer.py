"""Unit tests for Clizzer VisualAnalyzer."""

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

# Ensure project root is in sys.path for direct pytest invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
import pytest

import backend.core.visual_analyzer as va_module
from backend.core.visual_analyzer import VisualAnalyzer


def _create_synthetic_video(
    file_path: str,
    duration_sec: float = 3.0,
    fps: float = 30.0,
    with_cut: bool = True,
    static: bool = False,
) -> None:
    """Helper to generate a lightweight synthetic test video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    total_frames = int(duration_sec * fps)
    out = cv2.VideoWriter(file_path, fourcc, fps, (80, 80))

    cut_frame = int(total_frames / 2) if with_cut else total_frames + 1

    for i in range(total_frames):
        if static:
            frame = np.full((80, 80, 3), 128, dtype=np.uint8)
        elif i < cut_frame:
            # First scene: dark with subtle movement
            val = int(20 + (i % 5) * 5)
            frame = np.full((80, 80, 3), val, dtype=np.uint8)
        else:
            # Second scene: bright with distinct movement
            val = int(210 + (i % 10) * 4)
            frame = np.full((80, 80, 3), val, dtype=np.uint8)
        out.write(frame)

    out.release()


def test_init_validation():
    """Verify initialization parameters and validation."""
    analyzer = VisualAnalyzer(sample_interval=0.5)
    assert analyzer.sample_interval == 0.5

    custom = VisualAnalyzer(sample_interval=1.2)
    assert custom.sample_interval == 1.2

    with pytest.raises(ValueError):
        VisualAnalyzer(sample_interval=0)

    with pytest.raises(ValueError):
        VisualAnalyzer(sample_interval=-1.0)


def test_get_nearest_scene_boundary():
    """Verify nearest scene cut detection."""
    analyzer = VisualAnalyzer()
    scenes = [
        {"scene_index": 0, "start_time": 0.0, "end_time": 4.5},
        {"scene_index": 1, "start_time": 4.5, "end_time": 10.0},
        {"scene_index": 2, "start_time": 10.0, "end_time": 18.0},
    ]

    # Nearest boundary checks
    assert analyzer.get_nearest_scene_boundary(4.3, scenes) == 4.5
    assert analyzer.get_nearest_scene_boundary(0.1, scenes) == 0.0
    assert analyzer.get_nearest_scene_boundary(9.8, scenes) == 10.0
    assert analyzer.get_nearest_scene_boundary(17.5, scenes) == 18.0

    # Empty scene boundaries should return the original query timestamp
    assert analyzer.get_nearest_scene_boundary(7.25, []) == 7.25

    # Object attributes support (e.g. SimpleNamespace or Pydantic models)
    obj_scenes = [
        SimpleNamespace(start_time=0.0, end_time=6.0, scene_index=0),
        SimpleNamespace(start_time=6.0, end_time=12.0, scene_index=1),
    ]
    assert analyzer.get_nearest_scene_boundary(5.8, obj_scenes) == 6.0


def test_snap_to_scene():
    """Verify temporal snapping within tolerance."""
    analyzer = VisualAnalyzer()
    scenes = [
        {"scene_index": 0, "start_time": 0.0, "end_time": 5.0},
        {"scene_index": 1, "start_time": 5.0, "end_time": 15.0},
        {"scene_index": 2, "start_time": 15.0, "end_time": 30.0},
    ]

    # Within tolerance: should snap
    snapped = analyzer.snap_to_scene(4.6, 15.3, scenes, tolerance=1.0)
    assert snapped == (5.0, 15.0)

    # Outside tolerance: should remain unchanged
    unchanged = analyzer.snap_to_scene(3.0, 12.0, scenes, tolerance=1.0)
    assert unchanged == (3.0, 12.0)

    # Zero or negative tolerance should preserve timestamps
    no_tol = analyzer.snap_to_scene(4.8, 15.2, scenes, tolerance=0.0)
    assert no_tol == (4.8, 15.2)

    # Invalid input (start >= end) returned as-is
    inv = analyzer.snap_to_scene(20.0, 10.0, scenes)
    assert inv == (20.0, 10.0)

    # Over-snapping edge case (both snap to the same cut) should prevent collapsing
    collapsed_guard = analyzer.snap_to_scene(4.9, 5.1, scenes, tolerance=2.0)
    assert collapsed_guard[0] < collapsed_guard[1]


def test_analyze_file_not_found():
    """Verify FileNotFoundError on non-existent video path."""
    analyzer = VisualAnalyzer()
    with pytest.raises(FileNotFoundError):
        analyzer.analyze("non_existent_video_path.mp4")


def test_analyze_motion_static_and_dynamic():
    """Verify motion analysis produces normalized values."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        static_video_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        moving_video_path = f.name

    try:
        _create_synthetic_video(static_video_path, duration_sec=2.0, static=True)
        _create_synthetic_video(moving_video_path, duration_sec=2.0, with_cut=True)

        analyzer = VisualAnalyzer(sample_interval=0.5)

        # Static video motion should be all 0.0
        static_motion = analyzer._analyze_motion(static_video_path)
        assert len(static_motion) > 0
        for entry in static_motion:
            assert entry["motion_magnitude"] == 0.0

        # Moving video motion should be normalized to [0.0, 1.0] with peak at 1.0
        moving_motion = analyzer._analyze_motion(moving_video_path)
        assert len(moving_motion) > 0
        mags = [m["motion_magnitude"] for m in moving_motion]
        assert max(mags) == 1.0
        assert min(mags) == 0.0
        for m in moving_motion:
            assert "timestamp" in m
            assert 0.0 <= m["motion_magnitude"] <= 1.0
    finally:
        Path(static_video_path).unlink(missing_ok=True)
        Path(moving_video_path).unlink(missing_ok=True)


def test_histogram_fallback_scene_detection():
    """Verify histogram fallback detects scene transitions when PySceneDetect is unavailable."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        temp_path = f.name

    try:
        _create_synthetic_video(temp_path, duration_sec=3.0, fps=30.0, with_cut=True)
        analyzer = VisualAnalyzer(sample_interval=0.5)

        # Force fallback by temporarily disabling scenedetect flag
        original_flag = va_module._SCENEDETECT_AVAILABLE
        try:
            va_module._SCENEDETECT_AVAILABLE = False
            scenes = analyzer._detect_scenes(temp_path)
            assert len(scenes) >= 2
            # Cut at 1.5s should be detected within reasonable margin
            cut_near_1_5 = any(abs(s["end_time"] - 1.5) <= 0.6 for s in scenes)
            assert cut_near_1_5, f"Expected cut near 1.5s, got: {scenes}"
        finally:
            va_module._SCENEDETECT_AVAILABLE = original_flag
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_full_analysis_pipeline():
    """Verify full end-to-end analyze method output structure and progress reporting."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        temp_path = f.name

    try:
        _create_synthetic_video(temp_path, duration_sec=3.0, fps=30.0, with_cut=True)
        analyzer = VisualAnalyzer(sample_interval=0.5)

        progress_records: list[float] = []

        def on_progress(p: float) -> None:
            progress_records.append(p)

        result = analyzer.analyze(temp_path, progress_callback=on_progress)

        # Verify output keys
        assert "scene_boundaries" in result
        assert "motion_timeline" in result
        assert "frame_count" in result
        assert "fps" in result
        assert "duration" in result

        # Verify metadata
        assert result["frame_count"] == 90
        assert result["fps"] == 30.0
        assert np.isclose(result["duration"], 3.0, atol=0.1)

        # Verify scenes
        assert len(result["scene_boundaries"]) >= 1
        for scene in result["scene_boundaries"]:
            assert "start_time" in scene
            assert "end_time" in scene
            assert "scene_index" in scene
            assert scene["start_time"] <= scene["end_time"]

        # Verify motion timeline
        assert len(result["motion_timeline"]) > 0
        for motion in result["motion_timeline"]:
            assert "timestamp" in motion
            assert "motion_magnitude" in motion
            assert 0.0 <= motion["motion_magnitude"] <= 1.0

        # Verify progress callback
        assert len(progress_records) >= 3
        assert progress_records[0] == 0.0
        assert progress_records[-1] == 1.0
    finally:
        Path(temp_path).unlink(missing_ok=True)
