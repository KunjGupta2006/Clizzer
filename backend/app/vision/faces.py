from pathlib import Path
from typing import List, Optional, Tuple
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
from backend.app.models import FaceFrame, FaceTrack
from backend.app.audio.features import rolling_median_mad
from backend.app.logging import logger

_MEDIAPIPE_FACE_MESH = None
_CASCADE_FRONTAL = None
_CASCADE_PROFILE = None
_CASCADE_SMILE = None


def get_opencv_cascades():
    global _CASCADE_FRONTAL, _CASCADE_PROFILE, _CASCADE_SMILE
    if _CASCADE_FRONTAL is not None:
        return _CASCADE_FRONTAL, _CASCADE_PROFILE, _CASCADE_SMILE
    if cv2 is None:
        return None, None, None
    try:
        f_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        p_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        s_path = cv2.data.haarcascades + "haarcascade_smile.xml"
        _CASCADE_FRONTAL = cv2.CascadeClassifier(f_path)
        _CASCADE_PROFILE = cv2.CascadeClassifier(p_path)
        _CASCADE_SMILE = cv2.CascadeClassifier(s_path)
    except Exception as e:
        logger.warning("cascade_classifiers_failed_to_load", error=str(e))
    return _CASCADE_FRONTAL, _CASCADE_PROFILE, _CASCADE_SMILE


def get_mediapipe_mesh() -> Optional[object]:
    global _MEDIAPIPE_FACE_MESH
    if _MEDIAPIPE_FACE_MESH is not None:
        return _MEDIAPIPE_FACE_MESH
    try:
        import mediapipe as mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            _MEDIAPIPE_FACE_MESH = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=4,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
            return _MEDIAPIPE_FACE_MESH
    except Exception as e:
        logger.debug("mediapipe_not_available", error=str(e))
    return None


def extract_faces_from_proxy(
    proxy_path: Path,
    duration: float,
    source_width: int = 1920,
    source_height: int = 1080,
) -> Tuple[List[FaceTrack], np.ndarray, bool]:
    length = int(np.ceil(duration))
    face_score = np.zeros(length, dtype=np.float32)

    if cv2 is None or not proxy_path.exists():
        if cv2 is None:
            logger.warning("opencv_not_installed", fallback="no_face_tracks")
        return [], face_score, False

    cap = cv2.VideoCapture(str(proxy_path))
    if not cap.isOpened():
        return [], face_score, False

    fps = cap.get(cv2.CAP_PROP_FPS) or 2.0
    proxy_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    proxy_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)

    scale_x = source_width / max(1, proxy_w)
    scale_y = source_height / max(1, proxy_h)

    mesh = get_mediapipe_mesh()
    frame_idx = 0
    raw_tracks: List[FaceFrame] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t = frame_idx / fps
            if t > duration:
                break

            # CPU Optimization: Sample every 2nd frame (1 fps analysis) for 2x faster vision pass
            if frame_idx % 2 != 0:
                frame_idx += 1
                continue

            # Face detection: Mediapipe mesh or fast OpenCV Cascades (frontal + profile + smile)
            if mesh is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mesh.process(rgb)
                if results.multi_face_landmarks:
                    for lms in results.multi_face_landmarks:
                        xs = [p.x for p in lms.landmark]
                        ys = [p.y for p in lms.landmark]
                        min_x, max_x = min(xs) * proxy_w * scale_x, max(xs) * proxy_w * scale_x
                        min_y, max_y = min(ys) * proxy_h * scale_y, max(ys) * proxy_h * scale_y
                        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

                        upper_lip = lms.landmark[13].y
                        lower_lip = lms.landmark[14].y
                        mouth_open = float(np.clip((lower_lip - upper_lip) * 10.0, 0.0, 1.0))
                        expr_score = float(np.clip(mouth_open * 1.2, 0.0, 1.0))

                        raw_tracks.append(
                            FaceFrame(
                                t=t,
                                bbox=bbox,
                                mouth_open=mouth_open,
                                expr_score=expr_score,
                            )
                        )
            else:
                frontal, profile, smile = get_opencv_cascades()
                if frontal is not None and not frontal.empty():
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = frontal.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30))
                    if len(faces) == 0 and profile is not None and not profile.empty():
                        faces = profile.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30))

                    for (fx, fy, fw, fh) in faces:
                        bbox = [float(fx * scale_x), float(fy * scale_y), float(fw * scale_x), float(fh * scale_y)]
                        face_roi = gray[fy : fy + fh, fx : fx + fw]
                        mouth_open = 0.2
                        expr_score = 0.3
                        if smile is not None and not smile.empty() and face_roi.size > 0:
                            smiles = smile.detectMultiScale(face_roi, scaleFactor=1.2, minNeighbors=5, minSize=(15, 15))
                            if len(smiles) > 0:
                                mouth_open = 0.8
                                expr_score = 0.9

                        raw_tracks.append(
                            FaceFrame(
                                t=t,
                                bbox=bbox,
                                mouth_open=mouth_open,
                                expr_score=expr_score,
                            )
                        )
            frame_idx += 1
    finally:
        cap.release()

    if not raw_tracks:
        logger.info("no_faces_detected_in_video", fallback="weight_0")
        return [], face_score, False

    # Create FaceTrack
    track = FaceTrack(person_id=1, frames=raw_tracks)

    # Compute per-second face score: .5 expr_spike + .3 mouth_open + .2 face_present_change
    raw_expr = np.zeros(length, dtype=np.float32)
    raw_mouth = np.zeros(length, dtype=np.float32)
    raw_present = np.zeros(length, dtype=np.float32)

    for f in raw_tracks:
        sec = max(0, min(length - 1, int(np.floor(f.t))))
        raw_expr[sec] = max(raw_expr[sec], f.expr_score)
        raw_mouth[sec] = max(raw_mouth[sec], f.mouth_open)
        raw_present[sec] = 1.0

    present_change = np.zeros(length, dtype=np.float32)
    present_change[1:] = np.abs(np.diff(raw_present))

    face_score = 0.5 * raw_expr + 0.3 * raw_mouth + 0.2 * present_change
    norm_face = rolling_median_mad(face_score, window_size=120)
    return [track], norm_face, True
