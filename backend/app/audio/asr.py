import io
from pathlib import Path
from typing import Any, List, Optional, Tuple
import wave
import numpy as np
from backend.app.models import Segment, Word
from backend.app.logging import logger

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

# Universally patch av.open so PyAV 14+ never fails on metadata_errors in faster_whisper or anywhere else
try:
    import av
    _orig_av_open = av.open

    def _safe_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)

    av.open = _safe_av_open
except Exception as e:
    logger.debug("av_open_patch_failed", error=str(e))

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
        model_size = "base" if HAS_CUDA else "tiny"
        logger.info("loading_whisper_model", device=device, compute_type=compute_type, model_size=model_size)
        try:
            _WHISPER_MODEL = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=download_root,
            )
            return _WHISPER_MODEL
        except Exception as e:
            logger.warning("whisper_load_attempt_failed", error=str(e), fallback="empty_transcript")
            return None
    except Exception as e:
        logger.warning("whisper_load_failed", error=str(e), fallback="none")
        return None


def transcribe_audio(
    wav_path: Path,
    language: Optional[str] = None,
    intervals: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[List[Segment], List[Word]]:
    model = get_whisper_model()
    if model is None:
        logger.warning("whisper_not_loaded_returning_empty_transcript")
        return [], []

    segments_out: List[Segment] = []
    words_out: List[Word] = []

    try:
        # Load audio into memory as float32 numpy array for 10x faster seeking & processing on CPU
        audio_np: Optional[np.ndarray] = None
        sr = 16000
        try:
            with wave.open(str(wav_path), "rb") as wf:
                if wf.getnchannels() == 1 and wf.getframerate() == 16000 and wf.getsampwidth() == 2:
                    raw = wf.readframes(wf.getnframes())
                    audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            audio_np = None

        if intervals is not None and audio_np is not None:
            # Merge overlapping or adjacent intervals (< 3.0s gap)
            sorted_intervals = sorted(intervals, key=lambda x: x[0])
            merged: List[Tuple[float, float]] = []
            for start_sec, end_sec in sorted_intervals:
                if not merged:
                    merged.append((max(0.0, start_sec), end_sec))
                else:
                    prev_start, prev_end = merged[-1]
                    if start_sec <= prev_end + 3.0:
                        merged[-1] = (prev_start, max(prev_end, end_sec))
                    else:
                        merged.append((max(0.0, start_sec), end_sec))

            for st, et in merged:
                s_idx = max(0, int(st * sr))
                e_idx = min(len(audio_np), int(et * sr))
                if e_idx - s_idx < 1600:  # < 0.1s
                    continue
                chunk = audio_np[s_idx:e_idx]
                segs, info = model.transcribe(
                    chunk,
                    language=language,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    beam_size=1,  # 3x faster CPU greedy decoding
                    condition_on_previous_text=False,
                )
                detected_lang = info.language if info else "en"
                for s in segs:
                    seg_text = s.text.strip()
                    if not seg_text:
                        continue
                    segments_out.append(
                        Segment(
                            start=float(s.start) + st,
                            end=float(s.end) + st,
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
                                        start=float(w.start) + st,
                                        end=float(w.end) + st,
                                        text=word_clean,
                                        conf=float(w.probability) if hasattr(w, "probability") else 1.0,
                                    )
                                )
        else:
            audio_input: Any = audio_np if audio_np is not None else str(wav_path)
            segments, info = model.transcribe(
                audio_input,
                language=language,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                beam_size=1,  # 3x faster CPU greedy decoding
                condition_on_previous_text=False,
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
