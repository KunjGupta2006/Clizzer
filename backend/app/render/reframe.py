from typing import List, Tuple
from backend.app.models import FaceTrack


def build_reframe_filter(
    aspect: str,
    source_w: int,
    source_h: int,
    target_w: int = 1080,
    target_h: int = 1920,
    face_tracks: List[FaceTrack] | None = None,
    start_t: float = 0.0,
    end_t: float = 0.0,
) -> str:
    # Target dimensions based on aspect
    if aspect == "9:16":
        out_w, out_h = 1080, 1920
    elif aspect == "1:1":
        out_w, out_h = 1080, 1080
    elif aspect == "16:9":
        out_w, out_h = 1920, 1080
    else:
        out_w, out_h = target_w, target_h

    # Check for face tracking in the clip range for 9:16
    has_faces = False
    center_x = source_w / 2.0

    if aspect == "9:16" and face_tracks:
        clip_frames = []
        for track in face_tracks:
            for f in track.frames:
                if start_t <= f.t <= end_t:
                    clip_frames.append(f)

        if clip_frames:
            has_faces = True
            # EMA smoothing of face center x (alpha = 0.15)
            alpha = 0.15
            current_x = center_x
            for f in clip_frames:
                # f.bbox is [x, y, w, h]
                face_mid = f.bbox[0] + f.bbox[2] / 2.0
                current_x = alpha * face_mid + (1.0 - alpha) * current_x
            center_x = current_x

    if aspect == "9:16":
        if has_faces:
            # Crop 9:16 window centered on face
            crop_w = int(source_h * (9.0 / 16.0))
            # Ensure even dimensions
            crop_w = (crop_w // 2) * 2
            crop_x = int(center_x - crop_w / 2.0)
            # Clamp to frame bounds
            crop_x = max(0, min(source_w - crop_w, crop_x))
            crop_x = (crop_x // 2) * 2
            return f"crop={crop_w}:{source_h}:{crop_x}:0,scale={out_w}:{out_h}"
        else:
            # Blurred-background letterbox with gameplay centered
            return (
                f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
                f"crop={out_w}:{out_h},boxblur=20:5[bg];"
                f"[0:v]scale={out_w}:-2[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
            )

    if aspect == "1:1":
        return (
            f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},boxblur=20:5[bg];"
            f"[0:v]scale={out_w}:-2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
        )

    # 16:9 or default
    return f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2"
