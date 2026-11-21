from pathlib import Path
from typing import List, Tuple
import numpy as np
from backend.app.logging import logger

try:
    from scenedetect import SceneManager, open_video, ContentDetector, AdaptiveDetector
except ImportError:
    SceneManager = None


def detect_scene_cuts(video_path: Path, duration: float) -> List[float]:
    cut_timestamps: List[float] = []

    if SceneManager is not None and video_path.exists():
        try:
            video = open_video(str(video_path))
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=28.0))
            # Only add secondary adaptive detector on short videos to prevent 5-minute CPU stalls on long streams
            if duration <= 300.0:
                scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=3.0))
            scene_manager.detect_scenes(video=video)
            scene_list = scene_manager.get_scene_list()
            for scene in scene_list:
                cut_timestamps.append(scene[0].get_seconds())
            return cut_timestamps
        except Exception as e:
            logger.warning("scenedetect_failed", error=str(e), fallback="fast_frame_diff")

    return cut_timestamps


def compute_cut_density(cut_timestamps: List[float], duration: float, window_sec: float = 10.0) -> np.ndarray:
    length = int(np.ceil(duration))
    cut_density = np.zeros(length, dtype=np.float32)

    if not cut_timestamps:
        return cut_density

    cuts = np.array(cut_timestamps)
    half = window_sec / 2.0

    for sec in range(length):
        start = max(0.0, sec - half)
        end = min(duration, sec + half)
        count = np.sum((cuts >= start) & (cuts <= end))
        cut_density[sec] = float(count / window_sec)

    return cut_density
