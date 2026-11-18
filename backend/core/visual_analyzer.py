"""Visual Analysis Engine for Clizzer.

Provides video scene cut detection, frame-level motion timeline extraction,
and intelligent temporal snapping to scene boundaries.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Graceful optional import of PySceneDetect
try:
    from scenedetect import ContentDetector, detect

    _SCENEDETECT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    detect = None
    ContentDetector = None
    _SCENEDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)


class VisualAnalyzer:
    """Visual analysis engine for video scene cut detection and motion dynamics.

    Attributes:
        sample_interval: Seconds between analyzed frames during motion profiling.
    """

    def __init__(self, sample_interval: float = 0.5) -> None:
        """Initialize the VisualAnalyzer with a frame sampling interval.

        Args:
            sample_interval: Sampling interval in seconds between consecutive
                analyzed frames. Must be strictly positive. Defaults to 0.5s.

        Raises:
            ValueError: If sample_interval is less than or equal to zero.
        """
        if sample_interval <= 0:
            raise ValueError(
                f"sample_interval must be greater than 0, got {sample_interval}"
            )
        self.sample_interval = float(sample_interval)

    def analyze(
        self,
        video_path: str,
        progress_callback: Callable[[float], None] | None = None,
    ) -> dict[str, Any]:
        """Perform comprehensive visual analysis including scene detection and motion profiling.

        Args:
            video_path: Filesystem path to the input video file.
            progress_callback: Optional callable accepting a float between 0.0 and 1.0
                to report analysis progress.

        Returns:
            Dictionary containing:
                - scene_boundaries: List of dicts (start_time, end_time, scene_index)
                - motion_timeline: List of dicts (timestamp, motion_magnitude normalized 0-1)
                - frame_count: Total frame count in the video
                - fps: Video frame rate
                - duration: Total duration of the video in seconds

        Raises:
            FileNotFoundError: If the video file does not exist on disk.
            ValueError: If the video file cannot be opened or decoded.
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open or decode video file: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0 or np.isnan(fps):
            logger.warning("Invalid FPS reported by container for '%s'; defaulting to 30.0", video_path)
            fps = 30.0

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 0:
            frame_count = 0

        duration = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
        cap.release()

        if progress_callback and callable(progress_callback):
            progress_callback(0.0)

        # Stage 1: Detect scene cut boundaries
        logger.info("Starting scene cut detection for '%s'...", video_path)
        scene_boundaries = self._detect_scenes(str(path))

        if progress_callback and callable(progress_callback):
            progress_callback(0.25)

        # Stage 2: Analyze motion dynamics across consecutive frame samples
        logger.info("Starting motion analysis for '%s'...", video_path)

        def _motion_cb(p: float) -> None:
            if progress_callback and callable(progress_callback):
                progress_callback(0.25 + 0.75 * max(0.0, min(1.0, p)))

        motion_timeline = self._analyze_motion(
            str(path),
            progress_callback=_motion_cb if progress_callback else None,
        )

        # Infer duration / frame_count if container metadata was incomplete
        if duration <= 0.0 and motion_timeline:
            duration = motion_timeline[-1]["timestamp"]
        if frame_count <= 0 and duration > 0.0:
            frame_count = int(round(duration * fps))

        # Ensure scene_boundaries covers the full duration if empty
        if not scene_boundaries and duration > 0.0:
            scene_boundaries = [
                {
                    "start_time": 0.0,
                    "end_time": round(duration, 3),
                    "scene_index": 0,
                }
            ]

        if progress_callback and callable(progress_callback):
            progress_callback(1.0)

        return {
            "scene_boundaries": scene_boundaries,
            "motion_timeline": motion_timeline,
            "frame_count": frame_count,
            "fps": round(fps, 3),
            "duration": round(duration, 3),
        }

    def _detect_scenes(self, video_path: str) -> list[dict[str, Any]]:
        """Detect scene cut transitions in the video.

        Uses PySceneDetect if available; falls back gracefully to a color histogram
        correlation method if PySceneDetect is not installed or raises an error.

        Args:
            video_path: Path to the video file.

        Returns:
            List of dictionaries with keys:
                - start_time: Start timestamp of the scene in seconds
                - end_time: End timestamp of the scene in seconds
                - scene_index: Zero-based index of the detected scene
        """
        if _SCENEDETECT_AVAILABLE and detect is not None and ContentDetector is not None:
            try:
                scenes = detect(video_path, ContentDetector(threshold=27.0))
                if scenes:
                    scene_boundaries: list[dict[str, Any]] = []
                    for idx, (scene_start, scene_end) in enumerate(scenes):
                        start_sec = self._timecode_to_seconds(scene_start)
                        end_sec = self._timecode_to_seconds(scene_end)
                        scene_boundaries.append({
                            "start_time": round(float(start_sec), 3),
                            "end_time": round(float(end_sec), 3),
                            "scene_index": idx,
                        })
                    return scene_boundaries

                # No cuts detected by PySceneDetect: whole video is one scene
                duration = self._get_video_duration(video_path)
                return [
                    {
                        "start_time": 0.0,
                        "end_time": round(float(duration), 3),
                        "scene_index": 0,
                    }
                ]
            except Exception as exc:
                logger.warning(
                    "PySceneDetect failed on '%s': %s. Falling back to histogram scene detection.",
                    video_path,
                    exc,
                )
        else:
            logger.info(
                "scenedetect is unavailable; using color histogram comparison fallback."
            )

        return self._detect_scenes_histogram(video_path)

    def _detect_scenes_histogram(self, video_path: str) -> list[dict[str, Any]]:
        """Fallback scene detection using HSV histogram correlation between sampled frames.

        Args:
            video_path: Path to the video file.

        Returns:
            List of scene boundary dictionaries.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Could not open video file for histogram scene detection: %s", video_path)
            return []

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0 or np.isnan(fps):
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = (total_frames / fps) if fps > 0 and total_frames > 0 else 0.0

        # Sample at least twice per second or at self.sample_interval for responsiveness
        step = max(1, int(round(fps * min(self.sample_interval, 0.5))))

        prev_hist: np.ndarray | None = None
        cuts: list[float] = []
        frame_idx = 0

        while True:
            if frame_idx % step == 0:
                ret, frame = cap.read()
                if not ret:
                    break
                curr_time = frame_idx / fps

                # Calculate 2D HSV histogram (Hue and Saturation)
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
                else:
                    gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])

                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                if prev_hist is not None:
                    similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    # Significant drop in correlation indicates a scene cut
                    if similarity < 0.65:
                        # Prevent duplicate cut timestamps within 0.5s of each other
                        if not cuts or (curr_time - cuts[-1] >= 0.5):
                            cuts.append(curr_time)

                prev_hist = hist
            else:
                ret = cap.grab()
                if not ret:
                    break

            frame_idx += 1

        cap.release()

        # Build contiguous scene segments from cuts
        if duration <= 0.0 and frame_idx > 0:
            duration = frame_idx / fps

        split_points = [0.0] + [c for c in cuts if 0.0 < c < duration]
        if not split_points or split_points[-1] < duration:
            split_points.append(duration)

        # Deduplicate and sort split points
        split_points = sorted(list(set(round(p, 3) for p in split_points)))

        scene_boundaries: list[dict[str, Any]] = []
        for i in range(len(split_points) - 1):
            scene_boundaries.append({
                "start_time": split_points[i],
                "end_time": split_points[i + 1],
                "scene_index": i,
            })

        if not scene_boundaries:
            scene_boundaries = [
                {
                    "start_time": 0.0,
                    "end_time": round(float(duration), 3),
                    "scene_index": 0,
                }
            ]

        return scene_boundaries

    def _analyze_motion(
        self,
        video_path: str,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Compute frame-to-frame visual motion dynamics normalized between 0.0 and 1.0.

        Samples frames every sample_interval seconds. For each consecutive pair
        of sampled frames, calculates the mean absolute difference of grayscale pixel
        intensities, then normalizes all magnitudes across the timeline to [0, 1].

        Args:
            video_path: Filesystem path to the video file.
            progress_callback: Optional callable accepting progress fraction in [0.0, 1.0].

        Returns:
            List of dictionaries, each containing:
                - timestamp: Timestamp in seconds for the sampled frame pair
                - motion_magnitude: Normalized motion intensity between 0.0 and 1.0

        Raises:
            ValueError: If the video cannot be opened.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0 or np.isnan(fps):
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step_frames = max(1, int(round(fps * self.sample_interval)))

        raw_records: list[tuple[float, float]] = []
        prev_gray: np.ndarray | None = None
        frame_idx = 0

        while True:
            if frame_idx % step_frames == 0:
                ret, frame = cap.read()
                if not ret:
                    break
                curr_time = frame_idx / fps
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    mean_diff = float(np.mean(diff))
                    raw_records.append((round(curr_time, 3), mean_diff))

                prev_gray = gray
            else:
                ret = cap.grab()
                if not ret:
                    break

            frame_idx += 1

            if progress_callback and callable(progress_callback) and total_frames > 0:
                if frame_idx % (step_frames * 4 or 1) == 0:
                    progress_callback(min(0.99, frame_idx / total_frames))

        cap.release()

        if progress_callback and callable(progress_callback):
            progress_callback(1.0)

        if not raw_records:
            return []

        # Normalize magnitudes to [0.0, 1.0]
        magnitudes = [rec[1] for rec in raw_records]
        min_mag = min(magnitudes)
        max_mag = max(magnitudes)
        mag_range = max_mag - min_mag

        motion_timeline: list[dict[str, Any]] = []
        for ts, raw_val in raw_records:
            if mag_range > 1e-6:
                norm_mag = (raw_val - min_mag) / mag_range
            elif max_mag > 0.0:
                # Uniform non-zero motion: normalize against 8-bit max intensity
                norm_mag = min(1.0, raw_val / 255.0)
            else:
                norm_mag = 0.0

            motion_timeline.append({
                "timestamp": ts,
                "motion_magnitude": round(float(norm_mag), 4),
            })

        return motion_timeline

    def get_nearest_scene_boundary(
        self,
        timestamp: float,
        scene_boundaries: list[dict[str, Any]],
    ) -> float:
        """Find the nearest scene cut boundary timestamp to a given timestamp.

        Args:
            timestamp: Target query time in seconds.
            scene_boundaries: List of scene boundary dictionaries or objects
                containing start_time and end_time.

        Returns:
            float: Nearest scene cut boundary in seconds. Returns timestamp
            itself if scene_boundaries is empty.
        """
        if not scene_boundaries:
            return float(timestamp)

        boundary_times: set[float] = set()
        for scene in scene_boundaries:
            if isinstance(scene, dict):
                if "start_time" in scene:
                    boundary_times.add(float(scene["start_time"]))
                if "end_time" in scene:
                    boundary_times.add(float(scene["end_time"]))
            else:
                if hasattr(scene, "start_time"):
                    boundary_times.add(float(scene.start_time))
                if hasattr(scene, "end_time"):
                    boundary_times.add(float(scene.end_time))

        if not boundary_times:
            return float(timestamp)

        nearest = min(boundary_times, key=lambda cut: abs(cut - timestamp))
        return float(round(nearest, 3))

    def snap_to_scene(
        self,
        start: float,
        end: float,
        scene_boundaries: list[dict[str, Any]],
        tolerance: float = 3.0,
    ) -> tuple[float, float]:
        """Snap start and end timestamps to the nearest scene cut boundaries within tolerance.

        Args:
            start: Proposed clip start timestamp in seconds.
            end: Proposed clip end timestamp in seconds.
            scene_boundaries: List of scene boundary dictionaries.
            tolerance: Maximum allowed time deviation in seconds for snapping. Defaults to 3.0.

        Returns:
            tuple[float, float]: Snapped (start, end) timestamps.
        """
        if start >= end:
            return (float(round(start, 3)), float(round(end, 3)))

        if not scene_boundaries or tolerance <= 0:
            return (float(round(start, 3)), float(round(end, 3)))

        nearest_start = self.get_nearest_scene_boundary(start, scene_boundaries)
        cand_start = nearest_start if abs(nearest_start - start) <= tolerance else start

        nearest_end = self.get_nearest_scene_boundary(end, scene_boundaries)
        cand_end = nearest_end if abs(nearest_end - end) <= tolerance else end

        cand_start = max(0.0, cand_start)

        # Prevent inversion or zero-duration collapsing from over-snapping
        if cand_end <= cand_start:
            if cand_start < end:
                cand_end = end
            elif cand_end > start:
                cand_start = start
            else:
                cand_start = start
                cand_end = end

        return (float(round(cand_start, 3)), float(round(cand_end, 3)))

    def _get_video_duration(self, video_path: str) -> float:
        """Safely query video duration from OpenCV metadata."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps > 0.0 and frame_count > 0:
            return frame_count / fps
        return 0.0

    @staticmethod
    def _timecode_to_seconds(tc: Any) -> float:
        """Safely convert PySceneDetect FrameTimecode or numeric value to seconds."""
        if hasattr(tc, "seconds"):
            return float(tc.seconds)
        if hasattr(tc, "get_seconds"):
            return float(tc.get_seconds())
        try:
            return float(tc)
        except (TypeError, ValueError):
            return 0.0
