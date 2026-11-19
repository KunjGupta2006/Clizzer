import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field
from backend.app.text.prompts import GTA6_SYSTEM_PROMPT, WINDOW_SCORING_PROMPT
from backend.app.text.llm import LLMProvider, GroqProvider
from backend.app.audio.features import rolling_median_mad
from backend.app.errors import LLMUnavailable
from backend.app.logging import logger


class WindowScore(BaseModel):
    id: int
    hook: float = Field(default=5.0, ge=0.0, le=10.0)
    payoff: float = Field(default=5.0, ge=0.0, le=10.0)
    standalone: float = Field(default=5.0, ge=0.0, le=10.0)
    quotable: float = Field(default=5.0, ge=0.0, le=10.0)
    novelty: float = Field(default=5.0, ge=0.0, le=10.0)
    reason: str = ""


class BatchScoringSchema(BaseModel):
    windows: List[WindowScore]


def compute_heuristic_text_score(
    windows: List[Dict[str, Any]],
    duration: float,
    keywords: Optional[List[str]] = None,
) -> np.ndarray:
    length = int(np.ceil(duration))
    raw_score = np.zeros(length, dtype=np.float32)

    if not windows:
        return raw_score

    kw_lower = [k.lower() for k in (keywords or [])]

    for win in windows:
        start_sec = max(0, min(length - 1, int(np.floor(win["start"]))))
        end_sec = max(0, min(length - 1, int(np.ceil(win["end"]))))
        text_lower = win["text"].lower()

        # Speech rate (approx words per second)
        dur = max(1.0, win["end"] - win["start"])
        word_count = len(text_lower.split())
        rate = word_count / dur

        # Keyword hits
        kw_hits = sum(1 for kw in kw_lower if kw in text_lower)
        score_val = float(rate * 0.5 + kw_hits * 1.5)

        for s in range(start_sec, end_sec + 1):
            raw_score[s] = max(raw_score[s], score_val)

    return raw_score


def score_text_windows(
    windows: List[Dict[str, Any]],
    duration: float,
    candidate_intervals: Optional[List[Tuple[float, float]]] = None,
    provider: Optional[LLMProvider] = None,
    keywords: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], bool]:
    length = int(np.ceil(duration))
    raw_scores = np.zeros(length, dtype=np.float32)

    if not windows:
        return raw_scores, [], False

    # Filter windows overlapping candidate intervals to save LLM quota
    windows_for_llm: List[Dict[str, Any]] = []
    if candidate_intervals:
        for win in windows:
            w_start, w_end = win["start"], win["end"]
            overlaps = any(not (w_end < c_start or w_start > c_end) for c_start, c_end in candidate_intervals)
            if overlaps:
                windows_for_llm.append(win)
    else:
        windows_for_llm = windows[:15]  # Cap max windows if no intervals provided

    llm_available = False
    scored_windows_map: Dict[int, WindowScore] = {}

    if provider is not None:
        try:
            # Batch up to 5 windows per call
            batch_size = 5
            for i in range(0, len(windows_for_llm), batch_size):
                batch = windows_for_llm[i : i + batch_size]
                prompt_input = [
                    {"id": w["id"], "text": w["text"][:300], "duration": w["duration"]}
                    for w in batch
                ]
                user_msg = WINDOW_SCORING_PROMPT.format(windows_json=json.dumps(prompt_input))
                resp = provider.complete(GTA6_SYSTEM_PROMPT, user_msg, schema=BatchScoringSchema)
                for ws in resp.get("windows", []):
                    # Clamp scores 0-10
                    hook = float(np.clip(ws.get("hook", 5.0), 0.0, 10.0))
                    payoff = float(np.clip(ws.get("payoff", 5.0), 0.0, 10.0))
                    standalone = float(np.clip(ws.get("standalone", 5.0), 0.0, 10.0))
                    quotable = float(np.clip(ws.get("quotable", 5.0), 0.0, 10.0))
                    novelty = float(np.clip(ws.get("novelty", 5.0), 0.0, 10.0))
                    scored_windows_map[ws["id"]] = WindowScore(
                        id=ws["id"],
                        hook=hook,
                        payoff=payoff,
                        standalone=standalone,
                        quotable=quotable,
                        novelty=novelty,
                        reason=ws.get("reason", ""),
                    )
            llm_available = True
        except LLMUnavailable as e:
            logger.warning("llm_unavailable_falling_back_to_heuristic", error=str(e))
            llm_available = False
        except Exception as e:
            logger.warning("llm_call_unexpected_error", error=str(e))
            llm_available = False

    # Spread window scores across seconds
    # Formula: text_score = mean(hook, payoff, quotable, novelty)/10*.8 + standalone/10*.2
    heuristic_fallback = compute_heuristic_text_score(windows, duration, keywords)

    for win in windows:
        start_sec = max(0, min(length - 1, int(np.floor(win["start"]))))
        end_sec = max(0, min(length - 1, int(np.ceil(win["end"]))))

        if win["id"] in scored_windows_map:
            ws = scored_windows_map[win["id"]]
            content_mean = (ws.hook + ws.payoff + ws.quotable + ws.novelty) / 4.0
            val = (content_mean / 10.0) * 0.8 + (ws.standalone / 10.0) * 0.2
            win["llm_score"] = float(val)
            win["reason"] = ws.reason
        else:
            val = float(heuristic_fallback[start_sec] if start_sec < len(heuristic_fallback) else 0.5)
            win["llm_score"] = val
            win["reason"] = "Heuristic text score"

        for s in range(start_sec, end_sec + 1):
            raw_scores[s] = max(raw_scores[s], val)

    norm_scores = rolling_median_mad(raw_scores, window_size=120)
    return norm_scores, windows, llm_available
