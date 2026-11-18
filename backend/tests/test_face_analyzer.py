"""Unit tests for FaceAnalyzer and FaceFrame in Clizzer."""

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
import pytest

from backend.core.face_analyzer import FaceAnalyzer, FaceFrame, _extract_point


@dataclass
class MockLandmarkPoint:
    x: float
    y: float
    z: float = 0.0


def _build_landmarks(points_dict: dict[int, tuple[float, float, float]]):
    """Build mock landmarks supporting both list/dict indexing and .landmark attribute."""
    # MediaPipe has 468+ landmarks
    max_idx = max(max(points_dict.keys(), default=0), 468)
    coords = [MockLandmarkPoint(0.5, 0.5, 0.0) for _ in range(max_idx + 1)]
    for idx, (x, y, z) in points_dict.items():
        coords[idx] = MockLandmarkPoint(x, y, z)
    return SimpleNamespace(landmark=coords)


def _create_synthetic_face_video(file_path: str, duration_sec: float = 2.0, fps: float = 10.0) -> None:
    """Generate a lightweight synthetic video with a drawn oval simulating a face."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    total_frames = int(duration_sec * fps)
    out = cv2.VideoWriter(file_path, fourcc, fps, (120, 120))

    for i in range(total_frames):
        frame = np.full((120, 120, 3), 40, dtype=np.uint8)
        # Draw a bright face-like oval in the center
        cv2.ellipse(frame, (60, 60), (25, 35), 0, 0, 360, (220, 200, 180), -1)
        # Draw eye spots
        cv2.circle(frame, (50, 50), 4, (40, 40, 40), -1)
        cv2.circle(frame, (70, 50), 4, (40, 40, 40), -1)
        # Draw mouth
        cv2.ellipse(frame, (60, 75), (10, 5), 0, 0, 180, (50, 50, 180), -1)
        out.write(frame)

    out.release()


def _create_blank_video(file_path: str, duration_sec: float = 1.0, fps: float = 10.0) -> None:
    """Generate a lightweight blank video with no faces."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    total_frames = int(duration_sec * fps)
    out = cv2.VideoWriter(file_path, fourcc, fps, (80, 80))

    for _ in range(total_frames):
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        out.write(frame)

    out.release()


class TestFaceFrame:
    """Test suite for FaceFrame dataclass."""

    def test_face_frame_creation(self):
        frame = FaceFrame(
            timestamp=1.5,
            bbox=(0.2, 0.2, 0.4, 0.4),
            center_x=0.4,
            center_y=0.4,
            mouth_open_ratio=0.25,
            smile_ratio=0.75,
            confidence=0.95,
        )
        assert frame.timestamp == 1.5
        assert frame.bbox == (0.2, 0.2, 0.4, 0.4)
        assert frame.center_x == 0.4
        assert frame.center_y == 0.4
        assert frame.mouth_open_ratio == 0.25
        assert frame.smile_ratio == 0.75
        assert frame.confidence == 0.95


class TestFaceAnalyzerInit:
    """Test suite for FaceAnalyzer initialization."""

    def test_default_init(self):
        analyzer = FaceAnalyzer()
        assert analyzer.sample_interval == 0.5
        assert analyzer.min_detection_confidence == 0.5
        # MediaPipe should NOT be initialized on __init__ (lazy init)
        assert analyzer._face_detection is None
        assert analyzer._face_mesh is None

    def test_custom_init(self):
        analyzer = FaceAnalyzer(sample_interval=1.0, min_detection_confidence=0.7)
        assert analyzer.sample_interval == 1.0
        assert analyzer.min_detection_confidence == 0.7

    def test_dict_config_init(self):
        cfg = {"sample_interval": 0.25, "min_detection_confidence": 0.8}
        analyzer = FaceAnalyzer(cfg)
        assert analyzer.sample_interval == 0.25
        assert analyzer.min_detection_confidence == 0.8

    def test_none_config_init(self):
        analyzer = FaceAnalyzer(None)
        assert analyzer.sample_interval == 0.5
        assert analyzer.min_detection_confidence == 0.5


class TestLandmarkCalculations:
    """Test suite for mouth open and smile calculations."""

    def test_extract_point_formats(self):
        # Attribute object
        pt1 = MockLandmarkPoint(0.1, 0.2, 0.3)
        assert _extract_point([pt1], 0) == (0.1, 0.2, 0.3)

        # Coordinate tuple
        pt2 = (0.4, 0.5, 0.6)
        assert _extract_point([pt2], 0) == (0.4, 0.5, 0.6)

        # Dict
        pt3 = {"x": 0.7, "y": 0.8, "z": 0.9}
        assert _extract_point([pt3], 0) == (0.7, 0.8, 0.9)

        # Container with .landmark
        container = SimpleNamespace(landmark=[pt1])
        assert _extract_point(container, 0) == (0.1, 0.2, 0.3)

    def test_calculate_mouth_open_closed(self):
        analyzer = FaceAnalyzer()
        # Closed mouth: landmark 13 and 14 very close vertically
        # Corners (78, 308) width = 0.14
        lm = _build_landmarks({
            13: (0.5, 0.600, 0.0),
            14: (0.5, 0.602, 0.0),  # v_dist = 0.002
            78: (0.43, 0.601, 0.0),
            308: (0.57, 0.601, 0.0), # h_dist = 0.14
        })
        ratio = analyzer._calculate_mouth_open(lm)
        assert 0.0 <= ratio < 0.05

    def test_calculate_mouth_open_wide(self):
        analyzer = FaceAnalyzer()
        # Open mouth: vertical separation 0.10, corners width 0.14
        lm = _build_landmarks({
            13: (0.5, 0.55, 0.0),
            14: (0.5, 0.67, 0.0),  # v_dist = 0.12
            78: (0.43, 0.60, 0.0),
            308: (0.57, 0.60, 0.0), # h_dist = 0.14
        })
        ratio = analyzer._calculate_mouth_open(lm)
        assert 0.7 <= ratio <= 1.0

    def test_calculate_mouth_open_zero_division(self):
        analyzer = FaceAnalyzer()
        # Overlapping corners (h_dist = 0)
        lm = _build_landmarks({
            13: (0.5, 0.55, 0.0),
            14: (0.5, 0.65, 0.0),
            78: (0.5, 0.60, 0.0),
            308: (0.5, 0.60, 0.0),
        })
        assert analyzer._calculate_mouth_open(lm) == 0.0

    def test_calculate_smile_neutral(self):
        analyzer = FaceAnalyzer()
        # Neutral face:
        # Nose (1) at (0.5, 0.40)
        # Lip corners (61, 291) at (0.44, 0.54) and (0.56, 0.54)
        # width = 0.12, d_nose = hypot(0.06, 0.14) = 0.152 -> ratio = 0.787
        lm = _build_landmarks({
            1: (0.50, 0.40, 0.0),
            61: (0.44, 0.54, 0.0),
            291: (0.56, 0.54, 0.0),
        })
        smile = analyzer._calculate_smile(lm)
        assert smile == 0.0  # Baseline neutral maps to 0.0

    def test_calculate_smile_broad(self):
        analyzer = FaceAnalyzer()
        # Broad smile: corners lift upward to y=0.48 and widen to width=0.20
        # d_nose = hypot(0.10, 0.08) = 0.128 -> ratio = 0.20 / 0.128 = 1.56
        lm = _build_landmarks({
            1: (0.50, 0.40, 0.0),
            61: (0.40, 0.48, 0.0),
            291: (0.60, 0.48, 0.0),
        })
        smile = analyzer._calculate_smile(lm)
        assert smile == 1.0  # Broad smile clamped to 1.0


class TestTrackingAndSmoothing:
    """Test suite for track smoothing and reframing center interpolation."""

    def test_smooth_track_empty(self):
        analyzer = FaceAnalyzer()
        assert analyzer._smooth_track([]) == []

    def test_smooth_track_single(self):
        analyzer = FaceAnalyzer()
        frames = [FaceFrame(0.5, (0.1, 0.1, 0.2, 0.2), 0.2, 0.2, 0.1, 0.1, 0.9)]
        track = analyzer._smooth_track(frames, window=5)
        assert len(track) == 1
        assert track[0]["timestamp"] == 0.5
        assert track[0]["center_x"] == 0.2
        assert track[0]["center_y"] == 0.2

    def test_smooth_track_moving_average(self):
        analyzer = FaceAnalyzer()
        # Jumpy face centers: [0.5, 0.5, 0.9, 0.5, 0.5]
        frames = [
            FaceFrame(0.0, (0.0, 0.0, 0.1, 0.1), 0.5, 0.5, 0.0, 0.0, 0.9),
            FaceFrame(0.5, (0.0, 0.0, 0.1, 0.1), 0.5, 0.5, 0.0, 0.0, 0.9),
            FaceFrame(1.0, (0.0, 0.0, 0.1, 0.1), 0.9, 0.9, 0.0, 0.0, 0.9),
            FaceFrame(1.5, (0.0, 0.0, 0.1, 0.1), 0.5, 0.5, 0.0, 0.0, 0.9),
            FaceFrame(2.0, (0.0, 0.0, 0.1, 0.1), 0.5, 0.5, 0.0, 0.0, 0.9),
        ]
        track = analyzer._smooth_track(frames, window=3)
        assert len(track) == 5
        # The peak at index 2 (0.9) smoothed with neighbors (0.5, 0.9, 0.5) / 3 = 0.6333
        assert 0.60 <= track[2]["center_x"] <= 0.65
        assert 0.60 <= track[2]["center_y"] <= 0.65

    def test_get_reframe_center_empty_track(self):
        analyzer = FaceAnalyzer()
        # When no face is detected, must return center (0.5, 0.5)
        assert analyzer.get_reframe_center(1.0, []) == (0.5, 0.5)

    def test_get_reframe_center_single_track(self):
        analyzer = FaceAnalyzer()
        track = [{"timestamp": 1.0, "center_x": 0.35, "center_y": 0.45}]
        assert analyzer.get_reframe_center(0.0, track) == (0.35, 0.45)
        assert analyzer.get_reframe_center(2.0, track) == (0.35, 0.45)

    def test_get_reframe_center_interpolation(self):
        analyzer = FaceAnalyzer()
        track = [
            {"timestamp": 0.0, "center_x": 0.20, "center_y": 0.30},
            {"timestamp": 1.0, "center_x": 0.60, "center_y": 0.70},
        ]
        # Exact boundary matches
        assert analyzer.get_reframe_center(0.0, track) == (0.20, 0.30)
        assert analyzer.get_reframe_center(1.0, track) == (0.60, 0.70)

        # Midpoint interpolation (t=0.5 -> center (0.40, 0.50))
        mid = analyzer.get_reframe_center(0.5, track)
        assert pytest.approx(mid[0], 0.01) == 0.40
        assert pytest.approx(mid[1], 0.01) == 0.50

        # Beyond boundaries
        assert analyzer.get_reframe_center(-0.5, track) == (0.20, 0.30)
        assert analyzer.get_reframe_center(2.0, track) == (0.60, 0.70)


class TestEdgeCasesAndFallbacks:
    """Test suite for error handling, corrupt frames, and fallback detectors."""

    def test_detect_faces_corrupt_frame(self):
        analyzer = FaceAnalyzer()
        assert analyzer._detect_faces(None, 0.0) == []
        assert analyzer._detect_faces(np.array([]), 0.0) == []
        assert analyzer._detect_faces(np.zeros((0, 0, 3), dtype=np.uint8), 0.0) == []

    def test_mediapipe_import_error_graceful_handling(self):
        analyzer = FaceAnalyzer()
        # Simulate MediaPipe missing / raising ImportError
        with patch.dict(sys.modules, {"mediapipe": None}):
            # Should not crash, returns fallback or empty
            analyzer._mp_initialized = False
            analyzer._face_detection = None
            analyzer._face_mesh = None
            faces = analyzer._detect_faces(np.zeros((100, 100, 3), dtype=np.uint8), 0.0)
            assert isinstance(faces, list)

    def test_analyze_nonexistent_file(self):
        analyzer = FaceAnalyzer()
        result = analyzer.analyze("nonexistent_path_to_video.mp4")
        assert result["face_frames"] == []
        assert result["primary_face_track"] == []
        assert result["emotion_timeline"] == []
        assert result["face_screen_time"] == 0.0

    def test_analyze_blank_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "blank.mp4")
            _create_blank_video(video_path, duration_sec=1.0, fps=10.0)

            analyzer = FaceAnalyzer(sample_interval=0.5)
            progress_values = []
            result = analyzer.analyze(video_path, progress_callback=progress_values.append)

            assert "face_frames" in result
            assert "primary_face_track" in result
            assert "emotion_timeline" in result
            assert "face_screen_time" in result
            assert result["face_screen_time"] == 0.0
            assert result["primary_face_track"] == []
            # Progress callback should be called and complete with 1.0
            assert len(progress_values) > 0
            assert progress_values[-1] == 1.0

            # Reframe center on blank video must return default (0.5, 0.5)
            assert analyzer.get_reframe_center(0.5, result["primary_face_track"]) == (0.5, 0.5)

    def test_context_manager_and_close(self):
        with FaceAnalyzer() as analyzer:
            mock_fd = MagicMock()
            mock_fm = MagicMock()
            analyzer._face_detection = mock_fd
            analyzer._face_mesh = mock_fm
        # Exiting context calls close()
        mock_fd.close.assert_called_once()
        mock_fm.close.assert_called_once()
        assert analyzer._face_detection is None
        assert analyzer._face_mesh is None
        assert analyzer._is_closed is True

    def test_detect_faces_with_mocked_mediapipe(self):
        analyzer = FaceAnalyzer()
        mock_mesh = MagicMock()
        mock_detection = MagicMock()

        # Build simulated face mesh landmarks
        lm_obj = _build_landmarks({
            1: (0.50, 0.40, 0.0),    # Nose
            13: (0.50, 0.58, 0.0),   # Inner upper lip
            14: (0.50, 0.64, 0.0),   # Inner lower lip
            61: (0.42, 0.50, 0.0),   # Left outer lip corner
            78: (0.44, 0.60, 0.0),   # Left inner lip corner
            291: (0.58, 0.50, 0.0),  # Right outer lip corner
            308: (0.56, 0.60, 0.0),  # Right inner lip corner
        })

        mock_mesh.process.return_value = SimpleNamespace(multi_face_landmarks=[lm_obj])

        mock_det = SimpleNamespace(
            score=[0.92],
            location_data=SimpleNamespace(
                relative_bounding_box=SimpleNamespace(
                    xmin=0.35, ymin=0.30, width=0.30, height=0.40
                )
            ),
        )
        mock_detection.process.return_value = SimpleNamespace(detections=[mock_det])

        analyzer._face_mesh = mock_mesh
        analyzer._face_detection = mock_detection
        analyzer._mp_initialized = True
        analyzer._mp_available = True

        test_frame = np.full((100, 100, 3), 120, dtype=np.uint8)
        faces = analyzer._detect_faces(test_frame, timestamp=2.5)

        assert len(faces) == 1
        face = faces[0]
        assert face.timestamp == 2.5
        assert face.confidence == 0.92
        assert 0.0 <= face.center_x <= 1.0
        assert 0.0 <= face.center_y <= 1.0
        assert face.mouth_open_ratio > 0.0
        assert face.smile_ratio > 0.0

    def test_analyze_with_synthetic_video_and_mock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "face_video.mp4")
            _create_synthetic_face_video(video_path, duration_sec=1.0, fps=10.0)

            analyzer = FaceAnalyzer(sample_interval=0.2)
            # Inject mock detector that returns a face for every frame
            mock_mesh = MagicMock()
            lm_obj = _build_landmarks({
                1: (0.50, 0.40, 0.0),
                13: (0.50, 0.58, 0.0),
                14: (0.50, 0.64, 0.0),
                61: (0.42, 0.50, 0.0),
                78: (0.44, 0.60, 0.0),
                291: (0.58, 0.50, 0.0),
                308: (0.56, 0.60, 0.0),
            })
            mock_mesh.process.return_value = SimpleNamespace(multi_face_landmarks=[lm_obj])
            mock_det = MagicMock()
            mock_det.process.return_value = SimpleNamespace(detections=[])

            analyzer._face_mesh = mock_mesh
            analyzer._face_detection = mock_det
            analyzer._mp_initialized = True
            analyzer._mp_available = True

            res = analyzer.analyze(video_path)
            assert len(res["face_frames"]) > 0
            assert len(res["primary_face_track"]) > 0
            assert len(res["emotion_timeline"]) > 0
            assert res["face_screen_time"] == 100.0
            assert "mouth_open_ratio" in res["emotion_timeline"][0]
            assert "smile_ratio" in res["emotion_timeline"][0]
