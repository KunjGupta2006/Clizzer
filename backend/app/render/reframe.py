from typing import Any, Dict, List, Optional, Tuple
from backend.app.models import FaceTrack


def _parse_crop_box(
    region: Optional[Dict[str, Any]],
    def_x: float,
    def_y: float,
    def_w: float,
    def_h: float,
    source_w: int,
    source_h: int,
) -> Tuple[int, int, int, int]:
    """Parse normalized (0.0-1.0) or pixel coordinates into valid even-pixel bounds."""
    rx = float(region.get("x", def_x)) if region else def_x
    ry = float(region.get("y", def_y)) if region else def_y
    rw = float(region.get("w", def_w)) if region else def_w
    rh = float(region.get("h", def_h)) if region else def_h

    # Convert normalized (<= 1.0) to pixel dimensions
    if 0.0 <= rx <= 1.0 and 0.0 <= ry <= 1.0 and rw <= 1.0 and rh <= 1.0:
        px = int(round(rx * source_w))
        py = int(round(ry * source_h))
        pw = int(round(rw * source_w))
        ph = int(round(rh * source_h))
    else:
        px = int(round(rx))
        py = int(round(ry))
        pw = int(round(rw))
        ph = int(round(rh))

    px = max(0, min(source_w - 2, px))
    py = max(0, min(source_h - 2, py))
    pw = max(2, min(source_w - px, pw))
    ph = max(2, min(source_h - py, ph))

    # Snap to even dimensions for FFmpeg codecs
    pw = (pw // 2) * 2
    ph = (ph // 2) * 2
    px = (px // 2) * 2
    py = (py // 2) * 2
    return px, py, pw, ph


def build_reframe_filter(
    aspect: str,
    source_w: int,
    source_h: int,
    target_w: int = 1080,
    target_h: int = 1920,
    face_tracks: List[FaceTrack] | None = None,
    start_t: float = 0.0,
    end_t: float = 0.0,
    layout: str = "single",
    regions: Optional[List[Dict[str, Any]]] = None,
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

    norm_layout = (layout or "single").lower().replace("-", "_")

    # DUAL-FRAME STREAM COMPOSITING (Gaming Streamer + Gameplay)
    if aspect == "9:16" and (norm_layout in ("game_top", "cam_top", "top_bottom", "bottom_top", "pip") or (regions and len(regions) >= 2)):
        # Normalize layout aliases
        if norm_layout == "top_bottom":
            norm_layout = "game_top"
        elif norm_layout == "bottom_top":
            norm_layout = "cam_top"
        elif norm_layout not in ("game_top", "cam_top", "pip"):
            norm_layout = "game_top"

        # Identify camera and game regions
        cam_reg = None
        game_reg = None
        if regions:
            for r in regions:
                lbl = str(r.get("label", "")).lower()
                if any(k in lbl for k in ("cam", "face", "streamer", "webcam", "creator")):
                    cam_reg = r
                elif any(k in lbl for k in ("game", "play", "screen", "footage", "clip")):
                    game_reg = r
            if not cam_reg and len(regions) > 0:
                cam_reg = regions[0]
            if not game_reg and len(regions) > 1:
                game_reg = regions[1]

        # Extract coordinates with sensible stream defaults (full screen gameplay, top-right streamer cam)
        gx, gy, gw, gh = _parse_crop_box(game_reg, 0.0, 0.0, 1.0, 1.0, source_w, source_h)
        cx, cy, cw, ch = _parse_crop_box(cam_reg, 0.72, 0.05, 0.26, 0.32, source_w, source_h)

        # Smart exclusion: If gameplay is full-screen, automatically crop gameplay so the streamer webcam
        # does NOT appear duplicated inside the gameplay frame!
        is_default_game = not game_reg or (gx == 0 and gy == 0 and gw >= source_w - 4 and gh >= source_h - 4)
        if is_default_game and cam_reg:
            if cx > source_w * 0.45:
                # Camera is in right section; crop game to left portion
                gw = min(gw, cx)
                gw = (gw // 2) * 2
            elif cx + cw < source_w * 0.55:
                # Camera is in left section; crop game to right portion
                new_gx = cx + cw
                gw = source_w - new_gx
                gx = (new_gx // 2) * 2
                gw = (gw // 2) * 2

        half_h = out_h // 2  # 960

        # High-Quality slot scaling: preserves crisp native sharpness with aesthetic dark padding (no pixelated over-zoom)
        game_filter = (
            f"crop={gw}:{gh}:{gx}:{gy},scale={out_w}:{half_h}:force_original_aspect_ratio=decrease,"
            f"pad={out_w}:{half_h}:(ow-iw)/2:(oh-ih)/2:color=0x0a0a0f"
        )
        cam_filter = (
            f"crop={cw}:{ch}:{cx}:{cy},scale={out_w}:{half_h}:force_original_aspect_ratio=decrease,"
            f"pad={out_w}:{half_h}:(ow-iw)/2:(oh-ih)/2:color=0x0a0a0f"
        )

        if norm_layout == "game_top":
            # Game on top half (1080x960), Streamer Cam on bottom half (1080x960)
            return (
                f"[0:v]split=2[in_g][in_c];"
                f"[in_g]{game_filter}[top_slot];"
                f"[in_c]{cam_filter}[bottom_slot];"
                f"[top_slot][bottom_slot]vstack=inputs=2"
            )

        elif norm_layout == "cam_top":
            # Streamer Cam on top half (1080x960), Game on bottom half (1080x960)
            return (
                f"[0:v]split=2[in_c][in_g];"
                f"[in_c]{cam_filter}[top_slot];"
                f"[in_g]{game_filter}[bottom_slot];"
                f"[top_slot][bottom_slot]vstack=inputs=2"
            )

        elif norm_layout == "pip":
            # Picture-in-Picture: Base game in 1080x1920 + sharp streamer camera overlay
            bg_w = (out_w // 4 // 2) * 2
            bg_h = (out_h // 4 // 2) * 2
            return (
                f"[0:v]split=3[in_bg][in_game][in_pip];"
                f"[in_bg]scale={bg_w}:{bg_h}:force_original_aspect_ratio=increase,crop={bg_w}:{bg_h},boxblur=5:2,scale={out_w}:{out_h}[pip_bg];"
                f"[in_game]crop={gw}:{gh}:{gx}:{gy},scale={out_w}:-2[pip_fg];"
                f"[pip_bg][pip_fg]overlay=(W-w)/2:(H-h)/2[base_canvas];"
                f"[in_pip]crop={cw}:{ch}:{cx}:{cy},scale=360:270:force_original_aspect_ratio=decrease,"
                f"pad=360:270:(ow-iw)/2:(oh-ih)/2:color=0x0a0a0f,drawbox=x=0:y=0:w=360:h=270:color=white@0.85:t=3[pip_cam];"
                f"[base_canvas][pip_cam]overlay=W-w-36:H-h-240"
            )

    # STANDARD SINGLE REFRAME (Face tracking or full screen letterbox with aesthetic blur)
    if aspect == "9:16":
        if abs((source_w / max(1, source_h)) - (9.0 / 16.0)) < 0.05:
            return f"scale={out_w}:{out_h}"

        bg_w = (out_w // 4 // 2) * 2
        bg_h = (out_h // 4 // 2) * 2
        return (
            f"[0:v]scale={bg_w}:{bg_h}:force_original_aspect_ratio=increase,"
            f"crop={bg_w}:{bg_h},boxblur=5:2,scale={out_w}:{out_h}[bg];"
            f"[0:v]scale={out_w}:-2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
        )

    if aspect == "1:1":
        bg_w = (out_w // 4 // 2) * 2
        bg_h = (out_h // 4 // 2) * 2
        return (
            f"[0:v]scale={bg_w}:{bg_h}:force_original_aspect_ratio=increase,"
            f"crop={bg_w}:{bg_h},boxblur=5:2,scale={out_w}:{out_h}[bg];"
            f"[0:v]scale={out_w}:-2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
        )

    # 16:9 or default
    return f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2"
