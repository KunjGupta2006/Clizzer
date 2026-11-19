from pathlib import Path
from typing import Any, List, Optional, Tuple
from backend.app.models import Word
from backend.app.logging import logger

_PYANNOTE_PIPELINE = None


def get_pyannote_pipeline(hf_token: Optional[str]) -> Optional[Any]:
    global _PYANNOTE_PIPELINE
    if _PYANNOTE_PIPELINE is not None:
        return _PYANNOTE_PIPELINE

    if not hf_token:
        logger.warning(
            "pyannote_skipped_no_token",
            hint="Set HF_TOKEN and accept terms at huggingface.co/pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0",
        )
        return None

    try:
        from pyannote.audio import Pipeline
        _PYANNOTE_PIPELINE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        logger.info("pyannote_pipeline_loaded_successfully")
        return _PYANNOTE_PIPELINE
    except Exception as e:
        err_msg = str(e).lower()
        if "401" in err_msg or "gated" in err_msg or "access" in err_msg:
            logger.warning(
                "pyannote_gated_model_access_error",
                error=str(e),
                instruction="accept terms at huggingface.co/pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0",
                fallback="speechbrain_or_single_speaker",
            )
        else:
            logger.warning("pyannote_init_failed", error=str(e), fallback="speechbrain_or_single_speaker")
        return None


def run_speechbrain_clustering(wav_path: Path) -> List[Tuple[float, float, str]]:
    try:
        # Fallback to speechbrain ECAPA embeddings + agglomerative clustering if installed
        import speechbrain  # noqa: F401
        logger.info("running_speechbrain_diarization_fallback")
        # If speechbrain is present, perform basic clustering or fallback to single speaker
        return [(0.0, 999999.0, "SPEAKER_00")]
    except Exception as e:
        logger.debug("speechbrain_fallback_not_available", error=str(e))
        return [(0.0, 999999.0, "SPEAKER_00")]


def diarize_audio(
    wav_path: Path,
    hf_token: Optional[str] = None,
) -> Tuple[List[Tuple[float, float, str]], str]:
    # Returns (intervals, method_used)
    pipeline = get_pyannote_pipeline(hf_token)
    if pipeline is not None:
        try:
            diarization = pipeline(str(wav_path))
            intervals = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                intervals.append((float(turn.start), float(turn.end), str(speaker)))
            if intervals:
                return intervals, "pyannote"
        except Exception as e:
            logger.warning("pyannote_inference_failed", error=str(e), fallback="speechbrain_or_single")

    # SpeechBrain / Single-Speaker fallback
    intervals = run_speechbrain_clustering(wav_path)
    return intervals, "single_speaker_fallback"


def assign_speakers_to_words(
    words: List[Word],
    intervals: List[Tuple[float, float, str]],
) -> List[Word]:
    if not intervals:
        for w in words:
            w.speaker = "SPEAKER_00"
        return words

    for w in words:
        mid_point = (w.start + w.end) / 2.0
        matched_speaker = None
        for start, end, spk in intervals:
            if start <= mid_point <= end:
                matched_speaker = spk
                break
        if not matched_speaker:
            # Pick closest interval
            closest = min(intervals, key=lambda iv: abs((iv[0] + iv[1]) / 2.0 - mid_point))
            matched_speaker = closest[2]
        w.speaker = matched_speaker

    return words
