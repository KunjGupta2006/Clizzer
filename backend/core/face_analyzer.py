"""Face Analysis Engine for Clizzer.

Provides face detection, 468-point face mesh analysis (mouth open and smile ratios),
primary face tracking for 9:16 vertical video reframing, and emotion timeline extraction.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceFrame:
    """Represents face detection and facial expression metrics for a video frame.

    Attributes:
        timestamp: Time in seconds within the video where the face was detected.
        bbox: Bounding box tuple (x, y, w, h) in normalized coordinates (0.0 to 1.0).
        center_x: Horizontal center position of the face (normalized 0.0 to 1.0).
        center_y: Vertical center position of the face (normalized 0.0 to 1.0).
        mouth_open_ratio: Mouth aspect ratio measuring openness (0.0 to 1.0).
        smile_ratio: Ratio representing smile intensity (0.0 to 1.0).
        confidence: Detection or tracking confidence score (0.0 to 1.0).
    """

    timestamp: float
    bbox: tuple[float, float, float, float]
    center_x: float
    center_y: float
    mouth_open_ratio: float
    smile_ratio: float
    confidence: float


def _extract_point(landmarks: Any, index: int) -> tuple[float, float, float]:
    """Extract (x, y, z) coordinates for a given landmark index from various formats.

    Supports MediaPipe NormalizedLandmarkList, sequences of objects with .x/.y/.z attributes,
    lists/tuples of coordinate tuples, and dictionaries.

    Args:
        landmarks: Landmark container or sequence.
        index: Target landmark integer index.

    Returns:
        tuple[float, float, float]: (x, y, z) coordinates normalized to [0, 1].

    Raises:
        IndexError: If index is out of bounds.
        TypeError: If landmark type cannot be parsed.
    """
    if hasattr(landmarks, "landmark"):
        lm = landmarks.landmark[index]
    elif isinstance(landmarks, (list, tuple)):
        lm = landmarks[index]
    elif isinstance(landmarks, dict):
        lm = landmarks[index]
    else:
        raise TypeError(f"Unsupported landmarks container type: {type(landmarks)}")

    if hasattr(lm, "x") and hasattr(lm, "y"):
        x = float(lm.x)
        y = float(lm.y)
        z = float(getattr(lm, "z", 0.0))
        return x, y, z
    if isinstance(lm, (list, tuple)):
        x = float(lm[0])
        y = float(lm[1])
        z = float(lm[2]) if len(lm) > 2 else 0.0
        return x, y, z
    if isinstance(lm, dict):
        x = float(lm.get("x", 0.0))
        y = float(lm.get("y", 0.0))
        z = float(lm.get("z", 0.0))
        return x, y, z

    raise TypeError(f"Unsupported landmark point structure: {type(lm)}")


class FaceAnalyzer:
    """Engine for face detection, expression estimation, and primary face reframing.

    Uses MediaPipe face_detection and face_mesh for high-fidelity facial landmarking.
    Automatically lazy-initializes resources and gracefully handles missing dependencies,
    corrupt video frames, and videos without faces.

    Attributes:
        sample_interval: Time interval in seconds between sampled video frames.
        min_detection_confidence: Confidence threshold for face detection.
    """

    def __init__(
        self,
        sample_interval: float | dict[str, Any] | None = 0.5,
        min_detection_confidence: float = 0.5,
    ) -> None:
        """Initialize the FaceAnalyzer.

        Args:
            sample_interval: Frame sampling interval in seconds (default 0.5), or config dict.
            min_detection_confidence: Minimum detection confidence score (default 0.5).
        """
        if isinstance(sample_interval, dict):
            cfg = sample_interval
            self.sample_interval = max(
                0.01, float(cfg.get("sample_interval", 0.5))
            )
            self.min_detection_confidence = max(
                0.0,
                min(
                    1.0,
                    float(
                        cfg.get(
                            "min_detection_confidence", min_detection_confidence
                        )
                    ),
                ),
            )
        elif sample_interval is None:
            self.sample_interval = 0.5
            self.min_detection_confidence = max(
                0.0, min(1.0, float(min_detection_confidence))
            )
        else:
            self.sample_interval = max(0.01, float(sample_interval))
            self.min_detection_confidence = max(
                0.0, min(1.0, float(min_detection_confidence))
            )

        # Lazy-init MediaPipe face_detection and face_mesh
        self._mp_initialized: bool = False
        self._mp_available: bool | None = None
        self._face_detection: Any = None
        self._face_mesh: Any = None
        self._cascade_classifier: Any = None
        self._is_closed: bool = False

    def _init_mediapipe(self) -> bool:
        """Lazy-initialize MediaPipe face detection and face mesh models.

        Handles MediaPipe import errors gracefully and sets availability flags.

        Returns:
            bool: True if MediaPipe models are available and ready, False otherwise.
        """
        if self._face_detection is not None and self._face_mesh is not None:
            return True

        if self._mp_initialized and self._mp_available is False:
            return False

        self._mp_initialized = True
        try:
            import mediapipe as mp

            # Support mediapipe solutions API
            if (
                hasattr(mp, "solutions")
                and hasattr(mp.solutions, "face_detection")
                and hasattr(mp.solutions, "face_mesh")
            ):
                self._face_detection = (
                    mp.solutions.face_detection.FaceDetection(
                        min_detection_confidence=self.min_detection_confidence,
                        model_selection=1,
                    )
                )
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=5,
                    refine_landmarks=True,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=0.5,
                )
                self._mp_available = True
                logger.info(
                    "MediaPipe FaceDetection and FaceMesh initialized successfully."
                )
                return True

            logger.warning(
                "MediaPipe 'solutions' module is not present in the current installation."
            )
            self._mp_available = False
            return False

        except (ImportError, ModuleNotFoundError, AttributeError, Exception) as exc:
            logger.warning(
                "MediaPipe could not be loaded due to an import error: %s. "
                "Will use fallback face detection where possible.",
                exc,
            )
            self._mp_available = False
            return False

    def _init_fallback_detector(self) -> Any:
        """Lazy-initialize OpenCV Haar Cascade face detector as fallback."""
        if self._cascade_classifier is not None:
            return self._cascade_classifier
        try:
            cascade_path = (
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if os.path.exists(cascade_path):
                self._cascade_classifier = cv2.CascadeClassifier(cascade_path)
                return self._cascade_classifier
        except Exception as exc:
            logger.debug(
                "Failed to initialize OpenCV Haar cascade fallback: %s", exc
            )
        return None

    def _calculate_mouth_open(self, landmarks: Any) -> float:
        """Calculate mouth_open_ratio from landmarks 13,14 (lip top/bottom) vs 78,308 (corners).

        Args:
            landmarks: 468-point face mesh landmarks container.

        Returns:
            float: Mouth open ratio clamped between 0.0 (closed) and 1.0 (wide open).
        """
        try:
            p13 = _extract_point(landmarks, 13)
            p14 = _extract_point(landmarks, 14)
            p78 = _extract_point(landmarks, 78)
            p308 = _extract_point(landmarks, 308)

            # Vertical lip distance (landmarks 13 and 14)
            v_dist = math.hypot(p13[0] - p14[0], p13[1] - p14[1])
            # Horizontal mouth corner distance (landmarks 78 and 308)
            h_dist = math.hypot(p78[0] - p308[0], p78[1] - p308[1])

            if h_dist < 1e-6:
                return 0.0

            ratio = v_dist / h_dist
            # Clamp strictly to [0.0, 1.0]
            return float(max(0.0, min(1.0, ratio)))
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            logger.debug("Error calculating mouth_open_ratio: %s", exc)
            return 0.0

    def _calculate_smile(self, landmarks: Any) -> float:
        """Calculate smile_ratio from landmarks 61,291 (lip corners) relative to landmark 1 (nose).

        When smiling, lip corners (61, 291) pull upward toward the nose (landmark 1)
        and mouth width expands horizontally.

        Args:
            landmarks: 468-point face mesh landmarks container.

        Returns:
            float: Smile ratio clamped between 0.0 (neutral/frown) and 1.0 (broad smile).
        """
        try:
            p1 = _extract_point(landmarks, 1)  # Nose tip
            p61 = _extract_point(landmarks, 61)  # Outer left lip corner
            p291 = _extract_point(landmarks, 291)  # Outer right lip corner

            # Mouth width between outer corners
            width = math.hypot(p291[0] - p61[0], p291[1] - p61[1])

            # Distances from nose tip to both lip corners
            d_left = math.hypot(p61[0] - p1[0], p61[1] - p1[1])
            d_right = math.hypot(p291[0] - p1[0], p291[1] - p1[1])
            d_nose_corners = (d_left + d_right) / 2.0

            if d_nose_corners < 1e-6 or width < 1e-6:
                return 0.0

            # Scale and rotation-invariant ratio of mouth width to nose-corner distance
            raw_ratio = width / d_nose_corners

            # In neutral expression, ratio is typically ~0.80 - 0.95.
            # In broad smile, ratio elevates to ~1.30 - 1.50+.
            # Map baseline [0.85, 1.35] smoothly to [0.0, 1.0].
            smile = (raw_ratio - 0.85) / (1.35 - 0.85)
            return float(max(0.0, min(1.0, smile)))
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            logger.debug("Error calculating smile_ratio: %s", exc)
            return 0.0

    def _detect_faces(self, frame: np.ndarray, timestamp: float) -> list[FaceFrame]:
        """Detect faces in a video frame and extract facial metrics using MediaPipe or fallback.

        Args:
            frame: Video frame image as a BGR numpy array.
            timestamp: Timestamp in seconds for this frame.

        Returns:
            list[FaceFrame]: Detected face frames, or empty list if no face found.
        """
        # Handle corrupt or empty frame
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
            or frame.ndim < 2
        ):
            return []

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return []

        detected_faces: list[FaceFrame] = []

        # Attempt MediaPipe detection and mesh
        if self._init_mediapipe():
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False

                mesh_results = (
                    self._face_mesh.process(rgb_frame)
                    if self._face_mesh
                    else None
                )
                detection_results = (
                    self._face_detection.process(rgb_frame)
                    if self._face_detection
                    else None
                )

                # If face mesh produced landmarks
                if (
                    mesh_results is not None
                    and mesh_results.multi_face_landmarks
                ):
                    for (
                        face_landmarks
                    ) in mesh_results.multi_face_landmarks:
                        try:
                            # Derive normalized bounding box from mesh landmarks
                            xs = [lm.x for lm in face_landmarks.landmark]
                            ys = [lm.y for lm in face_landmarks.landmark]
                            xmin = max(0.0, min(1.0, min(xs)))
                            ymin = max(0.0, min(1.0, min(ys)))
                            xmax = max(0.0, min(1.0, max(xs)))
                            ymax = max(0.0, min(1.0, max(ys)))
                            bw = max(0.001, xmax - xmin)
                            bh = max(0.001, ymax - ymin)
                            cx = max(0.0, min(1.0, xmin + bw / 2.0))
                            cy = max(0.0, min(1.0, ymin + bh / 2.0))

                            mouth_open = self._calculate_mouth_open(
                                face_landmarks
                            )
                            smile = self._calculate_smile(face_landmarks)

                            # Match confidence with FaceDetection if available
                            confidence = self.min_detection_confidence
                            if (
                                detection_results
                                and detection_results.detections
                            ):
                                best_dist = float("inf")
                                for det in detection_results.detections:
                                    det_box = (
                                        det.location_data.relative_bounding_box
                                    )
                                    det_cx = det_box.xmin + det_box.width / 2.0
                                    det_cy = det_box.ymin + det_box.height / 2.0
                                    dist = math.hypot(cx - det_cx, cy - det_cy)
                                    if dist < best_dist:
                                        best_dist = dist
                                        if det.score:
                                            confidence = float(det.score[0])

                            detected_faces.append(
                                FaceFrame(
                                    timestamp=float(timestamp),
                                    bbox=(xmin, ymin, bw, bh),
                                    center_x=cx,
                                    center_y=cy,
                                    mouth_open_ratio=mouth_open,
                                    smile_ratio=smile,
                                    confidence=confidence,
                                )
                            )
                        except Exception as exc:
                            logger.debug("Error processing mesh face: %s", exc)

                # Fallback to FaceDetection bounding boxes if mesh returned no faces
                elif (
                    detection_results is not None
                    and detection_results.detections
                ):
                    for det in detection_results.detections:
                        try:
                            det_box = det.location_data.relative_bounding_box
                            xmin = max(0.0, min(1.0, float(det_box.xmin)))
                            ymin = max(0.0, min(1.0, float(det_box.ymin)))
                            bw = max(
                                0.001,
                                min(1.0 - xmin, float(det_box.width)),
                            )
                            bh = max(
                                0.001,
                                min(1.0 - ymin, float(det_box.height)),
                            )
                            cx = max(0.0, min(1.0, xmin + bw / 2.0))
                            cy = max(0.0, min(1.0, ymin + bh / 2.0))
                            conf = (
                                float(det.score[0])
                                if det.score
                                else self.min_detection_confidence
                            )

                            detected_faces.append(
                                FaceFrame(
                                    timestamp=float(timestamp),
                                    bbox=(xmin, ymin, bw, bh),
                                    center_x=cx,
                                    center_y=cy,
                                    mouth_open_ratio=0.0,
                                    smile_ratio=0.0,
                                    confidence=conf,
                                )
                            )
                        except Exception as exc:
                            logger.debug(
                                "Error processing detection face: %s", exc
                            )

                if detected_faces:
                    return detected_faces

            except Exception as exc:
                logger.warning(
                    "MediaPipe processing error on frame at timestamp %.2f: %s",
                    timestamp,
                    exc,
                )

        # Fallback to OpenCV Haar Cascade face detector
        cascade = self._init_fallback_detector()
        if cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(int(w * 0.05), int(h * 0.05)),
                )
                for x, y, fw, fh in rects:
                    nx = max(0.0, min(1.0, x / w))
                    ny = max(0.0, min(1.0, y / h))
                    nw = max(0.001, min(1.0 - nx, fw / w))
                    nh = max(0.001, min(1.0 - ny, fh / h))
                    cx = max(0.0, min(1.0, nx + nw / 2.0))
                    cy = max(0.0, min(1.0, ny + nh / 2.0))

                    detected_faces.append(
                        FaceFrame(
                            timestamp=float(timestamp),
                            bbox=(nx, ny, nw, nh),
                            center_x=cx,
                            center_y=cy,
                            mouth_open_ratio=0.0,
                            smile_ratio=0.0,
                            confidence=0.8,
                        )
                    )
            except Exception as exc:
                logger.debug("Haar cascade detection error: %s", exc)

        return detected_faces

    def _smooth_track(
        self, face_frames: list[Any], window: int = 5
    ) -> list[dict]:
        """Apply centered moving average smoothing to primary face center coordinates.

        Args:
            face_frames: Sequence of FaceFrame objects or dicts with center_x, center_y, timestamp.
            window: Smoothing window width (default 5).

        Returns:
            list[dict]: Smoothed track list of dicts: {"timestamp": t, "center_x": x, "center_y": y}.
        """
        if not face_frames:
            return []

        window = max(1, int(window))
        n = len(face_frames)

        # Extract timestamps, xs, ys safely from FaceFrame objects or dictionaries
        timestamps: list[float] = []
        xs: list[float] = []
        ys: list[float] = []

        for item in face_frames:
            if hasattr(item, "timestamp"):
                t = float(item.timestamp)
                x = float(item.center_x)
                y = float(item.center_y)
            elif isinstance(item, dict):
                t = float(item.get("timestamp", 0.0))
                x = float(item.get("center_x", item.get("x", 0.5)))
                y = float(item.get("center_y", item.get("y", 0.5)))
            else:
                continue
            timestamps.append(t)
            xs.append(x)
            ys.append(y)

        if not timestamps:
            return []

        if n == 1 or window == 1:
            return [
                {
                    "timestamp": round(timestamps[i], 4),
                    "center_x": round(max(0.0, min(1.0, xs[i])), 4),
                    "center_y": round(max(0.0, min(1.0, ys[i])), 4),
                }
                for i in range(len(timestamps))
            ]

        smoothed_track: list[dict] = []
        half_w = window // 2

        for i in range(n):
            w_start = max(0, i - half_w)
            w_end = min(n, i + half_w + 1)
            count = w_end - w_start

            avg_x = sum(xs[w_start:w_end]) / count
            avg_y = sum(ys[w_start:w_end]) / count

            smoothed_track.append(
                {
                    "timestamp": round(timestamps[i], 4),
                    "center_x": round(max(0.0, min(1.0, avg_x)), 4),
                    "center_y": round(max(0.0, min(1.0, avg_y)), 4),
                }
            )

        return smoothed_track

    def get_reframe_center(
        self, timestamp: float, primary_track: list[dict]
    ) -> tuple[float, float]:
        """Get the smoothed face center coordinates for video reframing at a given timestamp.

        Interpolates linearly between surrounding track points.
        Handles missing tracks or no face detected by defaulting to (0.5, 0.5).

        Args:
            timestamp: Target timestamp in seconds.
            primary_track: Smoothed face track list of dicts.

        Returns:
            tuple[float, float]: (center_x, center_y) normalized coordinates in [0.0, 1.0].
        """
        # Handle: no face detected (return center 0.5, 0.5)
        if not primary_track:
            return 0.5, 0.5

        if len(primary_track) == 1:
            item = primary_track[0]
            cx = float(item.get("center_x", 0.5))
            cy = float(item.get("center_y", 0.5))
            return max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))

        # Extract timestamps for binary search
        ts_list = [float(p.get("timestamp", 0.0)) for p in primary_track]

        # Boundary checks
        if timestamp <= ts_list[0]:
            first = primary_track[0]
            return float(first.get("center_x", 0.5)), float(
                first.get("center_y", 0.5)
            )

        if timestamp >= ts_list[-1]:
            last = primary_track[-1]
            return float(last.get("center_x", 0.5)), float(
                last.get("center_y", 0.5)
            )

        # Binary search for interval [i-1, i]
        idx = bisect.bisect_right(ts_list, timestamp)
        p0 = primary_track[idx - 1]
        p1 = primary_track[idx]

        t0 = float(p0.get("timestamp", 0.0))
        t1 = float(p1.get("timestamp", 0.0))
        dt = t1 - t0

        if dt < 1e-6:
            return float(p0.get("center_x", 0.5)), float(
                p0.get("center_y", 0.5)
            )

        # Linear interpolation
        alpha = (timestamp - t0) / dt
        x0 = float(p0.get("center_x", 0.5))
        y0 = float(p0.get("center_y", 0.5))
        x1 = float(p1.get("center_x", 0.5))
        y1 = float(p1.get("center_y", 0.5))

        interp_x = max(0.0, min(1.0, x0 + alpha * (x1 - x0)))
        interp_y = max(0.0, min(1.0, y0 + alpha * (y1 - y0)))

        return round(interp_x, 4), round(interp_y, 4)

    def analyze(
        self,
        video_path: str,
        progress_callback: Callable[[float], None] | None = None,
    ) -> dict[str, Any]:
        """Perform full face detection, expression profiling, and tracking across a video.

        Samples frames every sample_interval seconds using cv2.VideoCapture.
        Detects faces, computes mouth open and smile metrics, tracks the primary face,
        and aggregates face screen time percentage.

        Args:
            video_path: Path to the target video file.
            progress_callback: Optional callable invoked with completion ratio (0.0 to 1.0).

        Returns:
            dict with keys:
                face_frames: list[FaceFrame] of all detected faces across sampled frames.
                primary_face_track: list[dict] with smoothed centers for reframing.
                emotion_timeline: list[dict] with timestamp, mouth_open_ratio, smile_ratio.
                face_screen_time: float percentage of sampled frames containing a face (0-100%).
        """
        video_file = Path(video_path)
        if not video_file.is_file():
            logger.error("Video file does not exist: %s", video_path)
            if progress_callback is not None:
                progress_callback(1.0)
            return {
                "face_frames": [],
                "primary_face_track": [],
                "emotion_timeline": [],
                "face_screen_time": 0.0,
            }

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            logger.error("Failed to open video file with OpenCV: %s", video_path)
            if progress_callback is not None:
                progress_callback(1.0)
            return {
                "face_frames": [],
                "primary_face_track": [],
                "emotion_timeline": [],
                "face_screen_time": 0.0,
            }

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0.0 or math.isnan(fps):
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 0:
            total_frames = 0

        frame_step = max(1, int(round(fps * self.sample_interval)))

        all_face_frames: list[FaceFrame] = []
        primary_face_frames: list[FaceFrame] = []
        emotion_timeline: list[dict[str, float]] = []

        sampled_count = 0
        frames_with_face_count = 0
        current_frame_idx = 0
        prev_primary_center: tuple[float, float] | None = None
        consecutive_read_failures = 0

        try:
            while True:
                # Seek to target frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                ret, frame = cap.read()

                if not ret or frame is None:
                    consecutive_read_failures += 1
                    # Handle corrupt frames vs genuine end-of-file
                    if (
                        total_frames > 0
                        and current_frame_idx < total_frames - frame_step
                        and consecutive_read_failures < 10
                    ):
                        logger.warning(
                            "Corrupt or missing frame at index %d, skipping to next sample.",
                            current_frame_idx,
                        )
                        current_frame_idx += frame_step
                        continue
                    break

                consecutive_read_failures = 0
                timestamp = current_frame_idx / fps
                sampled_count += 1

                # Detect faces with corrupt frame protection
                try:
                    faces = self._detect_faces(frame, timestamp)
                except Exception as exc:
                    logger.warning(
                        "Error analyzing frame at index %d: %s. Skipping frame.",
                        current_frame_idx,
                        exc,
                    )
                    faces = []

                if faces:
                    frames_with_face_count += 1
                    all_face_frames.extend(faces)

                    # Track primary face (most frequent / largest / closest to previous)
                    if len(faces) == 1:
                        primary = faces[0]
                    else:
                        if prev_primary_center is None:
                            # Largest face by bounding box area
                            primary = max(
                                faces, key=lambda f: f.bbox[2] * f.bbox[3]
                            )
                        else:
                            px, py = prev_primary_center
                            # Score balances face area and proximity to previous center
                            primary = max(
                                faces,
                                key=lambda f: (f.bbox[2] * f.bbox[3])
                                * (
                                    1.0
                                    - 0.5
                                    * min(
                                        1.0,
                                        math.hypot(
                                            f.center_x - px, f.center_y - py
                                        ),
                                    )
                                ),
                            )

                    prev_primary_center = (primary.center_x, primary.center_y)
                    primary_face_frames.append(primary)

                    emotion_timeline.append(
                        {
                            "timestamp": round(primary.timestamp, 4),
                            "mouth_open_ratio": round(
                                primary.mouth_open_ratio, 4
                            ),
                            "smile_ratio": round(primary.smile_ratio, 4),
                        }
                    )

                # Report progress
                if progress_callback is not None:
                    try:
                        prog = (
                            min(1.0, current_frame_idx / total_frames)
                            if total_frames > 0
                            else 0.0
                        )
                        progress_callback(prog)
                    except Exception:
                        pass

                current_frame_idx += frame_step
                if total_frames > 0 and current_frame_idx >= total_frames:
                    break

        finally:
            cap.release()

        # Compute smoothed primary track and face screen time percentage
        primary_face_track = self._smooth_track(primary_face_frames, window=5)

        face_screen_time = (
            round((frames_with_face_count / sampled_count) * 100.0, 2)
            if sampled_count > 0
            else 0.0
        )

        if progress_callback is not None:
            try:
                progress_callback(1.0)
            except Exception:
                pass

        return {
            "face_frames": all_face_frames,
            "primary_face_track": primary_face_track,
            "emotion_timeline": emotion_timeline,
            "face_screen_time": face_screen_time,
        }

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._face_detection is not None:
            try:
                self._face_detection.close()
            except Exception as exc:
                logger.debug("Error releasing FaceDetection: %s", exc)
            self._face_detection = None

        if self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception as exc:
                logger.debug("Error releasing FaceMesh: %s", exc)
            self._face_mesh = None

        self._is_closed = True

    def __enter__(self) -> FaceAnalyzer:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Destructor cleanup."""
        self.close()
