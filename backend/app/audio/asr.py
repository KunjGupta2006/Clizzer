from pathlib import Path
from typing import Any, List, Optional, Tuple
from backend.app.models import Segment, Word
from backend.app.logging import logger

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

_WHISPER_MODEL = None


def get_whisper_model(download_root: Optional[str] = None) -> Optional[Any]:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    if WhisperModel is None:
        logger.warning("faster_whisper_not_available", fallback="heuristic_asr")
        return None

    try:
        device = "cuda" if HAS_CUDA else "cpu"
        compute_type = "float16" if HAS_CUDA else "int8"
        model_size = "large-v3" if HAS_CUDA else "small"
        logger.info("loading_whisper_model", device=device, compute_type=compute_type, model_size=model_size)
        # Attempt loading with local_files_only first to prevent hanging in offline/test environments
        try:
            _WHISPER_MODEL = WhisperModel(model_size, device=device, compute_type=compute_type, local_files_only=True)
            return _WHISPER_MODEL
        except Exception:
            _WHISPER_MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)
            return _WHISPER_MODEL
    except Exception as e:
        logger.warning("whisper_load_failed", error=str(e), fallback="cpu_int8_or_none")
        try:
            _WHISPER_MODEL = WhisperModel("tiny", device="cpu", compute_type="int8", local_files_only=True)
            return _WHISPER_MODEL
        except Exception:
            return None


def transcribe_audio(
    wav_path: Path,
    language: Optional[str] = None,
) -> Tuple[List[Segment], List[Word]]:
    model = get_whisper_model()
    if model is None:
        logger.warning("whisper_not_loaded_returning_empty_transcript")
        return [], []

    segments_out: List[Segment] = []
    words_out: List[Word] = []

    try:
        segments, info = model.transcribe(
            str(wav_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        detected_lang = info.language if info else "en"

        for s in segments:
            seg_text = s.text.strip()
            segments_out.append(
                Segment(
                    start=float(s.start),
                    end=float(s.end),
                    text=seg_text,
                    lang=detected_lang,
                )
            )
            if hasattr(s, "words") and s.words:
                for w in s.words:
                    word_clean = w.word.strip()
                    if word_clean:
                        words_out.append(
                            Word(
                                start=float(w.start),
                                end=float(w.end),
                                text=word_clean,
                                conf=float(w.probability) if hasattr(w, "probability") else 1.0,
                            )
                        )
    except Exception as e:
        logger.error("transcription_failed", wav=str(wav_path), error=str(e))

    return segments_out, words_out
