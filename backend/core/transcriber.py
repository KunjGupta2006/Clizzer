"""
Audio transcription module for Clizzer using faster-whisper.

Provides the Transcriber class for lazy-loaded speech-to-text conversion,
word-level timestamp extraction, and time-range word slicing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class Transcriber:
    """
    Transcribes audio files into timestamped text segments and words
    using faster-whisper.
    """

    def __init__(
        self,
        model_size: Union[str, Dict[str, Any], None] = "tiny",
        compute_type: str = "int8",
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Transcriber configuration.
        The underlying WhisperModel is lazy-loaded on first transcription or model access.

        Args:
            model_size: Model size name ('tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3')
                        or a config dictionary.
            compute_type: Computation type ('int8', 'float16', 'float32', etc.).
            device: Device to run model on ('cpu', 'cuda', 'auto').
            **kwargs: Extra configuration options.
        """
        if isinstance(model_size, dict):
            config = model_size
            self.model_size = str(config.get("model_size", "tiny"))
            self.compute_type = str(config.get("compute_type", compute_type))
            self.device = str(config.get("device", device))
        else:
            self.model_size = str(model_size) if model_size is not None else "tiny"
            self.compute_type = compute_type
            self.device = device

        self.extra_config = kwargs
        self._model = None

    @property
    def model(self) -> Any:
        """
        Lazy-loads and returns the faster_whisper.WhisperModel instance.
        """
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self) -> None:
        """
        Loads the faster-whisper model into memory.
        Raises ImportError if faster-whisper is not installed, or RuntimeError on failure.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "The 'faster-whisper' package is required for Transcriber. "
                "Install it using `pip install faster-whisper`."
            ) from exc

        try:
            logger.info(
                "Loading faster-whisper model '%s' (device=%s, compute_type=%s)...",
                self.model_size,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                model_size_or_path=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("faster-whisper model '%s' loaded successfully.", self.model_size)
        except Exception as exc:
            logger.error("Failed to load faster-whisper model: %s", exc)
            raise RuntimeError(
                f"Failed to load WhisperModel(model_size='{self.model_size}', "
                f"device='{self.device}', compute_type='{self.compute_type}'): {exc}"
            ) from exc

    def transcribe(
        self,
        audio_path: Union[str, Path],
        progress_callback: Optional[Callable[[float], None]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file using faster-whisper with word timestamps and VAD filtering.

        Args:
            audio_path: Path to the audio or video file.
            progress_callback: Optional callback callable receiving float progress (0.0 to 1.0).
            **kwargs: Optional additional keyword arguments to pass to model.transcribe.

        Returns:
            dict containing:
                - segments (list): List of segment dicts with 'id', 'start', 'end', 'text', 'words'.
                - full_text (str): Complete transcribed text combined.
                - language (str): Detected spoken language code.
                - language_probability (float): Detection confidence for the language.
                - duration (float): Total audio duration in seconds.
        """
        if not audio_path:
            raise ValueError("audio_path cannot be None or empty.")

        audio_str = str(audio_path)
        if not os.path.isfile(audio_str):
            raise FileNotFoundError(f"Audio file does not exist: {audio_str}")

        transcribe_params: Dict[str, Any] = {
            "beam_size": 5,
            "word_timestamps": True,
            "vad_filter": True,
        }
        transcribe_params.update(kwargs)

        try:
            segments_gen, info = self.model.transcribe(audio_str, **transcribe_params)
        except Exception as exc:
            logger.error("Transcription execution failed for '%s': %s", audio_str, exc)
            raise RuntimeError(f"Transcription failed for '{audio_str}': {exc}") from exc

        total_duration = float(getattr(info, "duration", 0.0) or 0.0)
        segments: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []

        try:
            for idx, segment in enumerate(segments_gen):
                seg_text = segment.text.strip() if hasattr(segment, "text") and segment.text else ""
                seg_start = round(float(segment.start), 3) if hasattr(segment, "start") else 0.0
                seg_end = round(float(segment.end), 3) if hasattr(segment, "end") else 0.0

                word_list: List[Dict[str, Any]] = []
                raw_words = getattr(segment, "words", None)
                if raw_words:
                    for w in raw_words:
                        w_text = w.word.strip() if hasattr(w, "word") and w.word else ""
                        w_start = round(float(w.start), 3) if hasattr(w, "start") else 0.0
                        w_end = round(float(w.end), 3) if hasattr(w, "end") else 0.0
                        word_dict: Dict[str, Any] = {
                            "word": w_text,
                            "start": w_start,
                            "end": w_end,
                        }
                        if hasattr(w, "probability") and w.probability is not None:
                            word_dict["probability"] = round(float(w.probability), 4)
                        word_list.append(word_dict)

                segment_dict: Dict[str, Any] = {
                    "id": getattr(segment, "id", idx),
                    "start": seg_start,
                    "end": seg_end,
                    "text": seg_text,
                    "words": word_list,
                }
                segments.append(segment_dict)

                if seg_text:
                    full_text_parts.append(seg_text)

                if progress_callback and callable(progress_callback):
                    try:
                        if total_duration > 0:
                            current_progress = min(1.0, max(0.0, seg_end / total_duration))
                        else:
                            current_progress = 0.0
                        progress_callback(current_progress)
                    except Exception as cb_exc:
                        logger.warning("Error in progress_callback: %s", cb_exc)

        except Exception as exc:
            logger.error("Error while iterating transcription segments: %s", exc)
            raise RuntimeError(f"Error reading transcription segments: {exc}") from exc

        if progress_callback and callable(progress_callback):
            try:
                progress_callback(1.0)
            except Exception:
                pass

        full_text = " ".join(full_text_parts).strip()
        detected_language = str(getattr(info, "language", "") or "")
        language_probability = round(float(getattr(info, "language_probability", 0.0) or 0.0), 4)

        return {
            "segments": segments,
            "full_text": full_text,
            "language": detected_language,
            "language_probability": language_probability,
            "duration": total_duration,
        }

    def get_words_in_range(
        self,
        segments: List[Dict[str, Any]],
        start: float,
        end: float,
        overlap: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Return all word dicts falling within the specified time range [start, end].

        Args:
            segments: List of segment dicts (containing 'words') or flat list of word dicts.
            start: Start timestamp in seconds.
            end: End timestamp in seconds.
            overlap: If True (default), includes words that overlap the time window.
                     If False, only includes words strictly contained within [start, end].

        Returns:
            list[dict]: All matching word dictionaries with 'word', 'start', and 'end'.
        """
        if not segments:
            return []

        range_start = min(float(start), float(end))
        range_end = max(float(start), float(end))

        # Flatten word candidates
        candidates: List[Dict[str, Any]] = []
        for item in segments:
            if not isinstance(item, dict):
                continue
            if "words" in item and isinstance(item["words"], list):
                candidates.extend(item["words"])
            elif "word" in item and "start" in item and "end" in item:
                candidates.append(item)

        results: List[Dict[str, Any]] = []
        for w in candidates:
            if not isinstance(w, dict) or "start" not in w or "end" not in w:
                continue

            try:
                w_start = float(w["start"])
                w_end = float(w["end"])
            except (ValueError, TypeError):
                continue

            if overlap:
                if range_start == range_end:
                    is_match = (w_start <= range_start <= w_end)
                else:
                    is_match = (w_end > range_start and w_start < range_end)
            else:
                is_match = (w_start >= range_start and w_end <= range_end)

            if is_match:
                results.append(dict(w))

        return results
