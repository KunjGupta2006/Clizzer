from pathlib import Path
from typing import List, Optional, Tuple
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
from backend.app.logging import logger

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_TOKENIZER = None

CLIP_LABELS = [
    "facecam only",
    "gameplay",
    "trailer footage",
    "menu or UI",
    "cutscene",
    "reaction overlay",
    "wanted level HUD",
    "vehicle chase",
    "explosion",
    "map screen",
]

ACTION_LABELS = ["gameplay", "cutscene", "vehicle chase", "explosion"]


def get_clip_model() -> Tuple[Optional[object], Optional[object], Optional[object]]:
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZER
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZER

    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        _CLIP_MODEL = model
        _CLIP_PREPROCESS = preprocess
        _CLIP_TOKENIZER = tokenizer
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZER
    except Exception as e:
        logger.debug("open_clip_not_available", error=str(e), fallback="heuristic_vision")
        return None, None, None


def classify_proxy_frames(
    proxy_path: Path,
    duration: float,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    # Returns (action_prob, trailer_prob, available)
    length = int(np.ceil(duration))
    action_prob = np.zeros(length, dtype=np.float32)
    trailer_prob = np.zeros(length, dtype=np.float32)

    if cv2 is None or not proxy_path.exists():
        return action_prob, trailer_prob, False

    model, preprocess, tokenizer = get_clip_model()
    if model is not None:
        try:
            import torch
            from PIL import Image

            text_tokens = tokenizer(CLIP_LABELS)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)

            cap = cv2.VideoCapture(str(proxy_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 2.0
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                t = frame_idx / fps
                sec = int(np.floor(t))
                if sec >= length:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                img_tensor = preprocess(pil_img).unsqueeze(0)

                with torch.no_grad():
                    image_features = model.encode_image(img_tensor)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    sim = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]

                # Map probabilities
                act_sum = 0.0
                for lbl in ACTION_LABELS:
                    act_sum += sim[CLIP_LABELS.index(lbl)]
                action_prob[sec] = max(action_prob[sec], float(act_sum))
                trailer_prob[sec] = max(trailer_prob[sec], float(sim[CLIP_LABELS.index("trailer footage")]))
                frame_idx += 1

            cap.release()
            return action_prob, trailer_prob, True
        except Exception as e:
            logger.warning("clip_inference_failed", error=str(e), fallback="heuristic_vision")

    # Heuristic fallback using motion / color variance
    cap = cv2.VideoCapture(str(proxy_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 2.0
    frame_idx = 0
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        sec = int(np.floor(t))
        if sec >= length:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion = float(np.mean(diff)) / 255.0
            action_prob[sec] = max(action_prob[sec], float(np.clip(motion * 3.0, 0.0, 1.0)))
        prev_gray = gray
        frame_idx += 1

    cap.release()
    return action_prob, trailer_prob, False
