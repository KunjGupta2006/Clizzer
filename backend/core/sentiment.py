"""
Sentiment and hype analysis module for Clizzer.

Provides SentimentAnalyzer for evaluating speech sentiment, extracting hype levels,
and scoring video segments for clip generation using NLTK VADER and heuristic boosters.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

DEFAULT_HYPE_WORDS: Sequence[str] = (
    # Explicitly requested terms
    "insane",
    "crazy",
    "unbelievable",
    "no way",
    "oh my god",
    "lets go",
    # Common variations and streaming/gaming hype keywords
    "let's go",
    "omg",
    "holy shit",
    "what the heck",
    "wtf",
    "hype",
    "hyped",
    "epic",
    "unreal",
    "clutch",
    "goat",
    "fire",
    "mindblowing",
    "mind blowing",
    "insanity",
    "legendary",
    "pog",
    "poggers",
    "huge",
    "w",
)


class SentimentAnalyzer:
    """
    Analyzes sentiment of transcription segments and computes hype scores
    using NLTK VADER combined with linguistic hype boosters.
    """

    def __init__(
        self,
        hype_words: Union[Sequence[str], Dict[str, Any], None] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the SentimentAnalyzer.
        The VADER SentimentIntensityAnalyzer is lazy-loaded on first use.

        Args:
            hype_words: Optional custom list of hype words/phrases, or config dict.
            config: Optional configuration dictionary.
            **kwargs: Extra parameters.
        """
        cfg = dict(config or {})
        if isinstance(hype_words, dict):
            cfg.update(hype_words)
            words = cfg.get("hype_words", None)
        else:
            words = hype_words or cfg.get("hype_words", None)

        self._vader = None
        self.config = cfg
        self.hype_words = list(words) if words is not None else list(DEFAULT_HYPE_WORDS)
        # Pre-sort hype words by length descending so longer multi-word phrases match first
        self.hype_words.sort(key=lambda s: len(s), reverse=True)

    @property
    def vader(self) -> Any:
        """
        Lazy-loads and returns the NLTK VADER SentimentIntensityAnalyzer instance.
        Auto-downloads 'vader_lexicon' if missing.
        """
        if self._vader is None:
            self._init_vader()
        return self._vader

    def _init_vader(self) -> None:
        """
        Initializes the VADER analyzer, automatically downloading vader_lexicon if necessary.
        """
        try:
            import nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
        except ImportError as exc:
            raise ImportError(
                "The 'nltk' package is required for SentimentAnalyzer. "
                "Install it using `pip install nltk`."
            ) from exc

        try:
            self._vader = SentimentIntensityAnalyzer()
        except LookupError:
            logger.info("NLTK 'vader_lexicon' resource not found. Auto-downloading...")
            try:
                nltk.download("vader_lexicon", quiet=True)
                self._vader = SentimentIntensityAnalyzer()
                logger.info("NLTK 'vader_lexicon' downloaded and VADER initialized successfully.")
            except Exception as dl_exc:
                logger.error("Failed to auto-download 'vader_lexicon': %s", dl_exc)
                raise RuntimeError(
                    f"Failed to auto-download required NLTK 'vader_lexicon': {dl_exc}"
                ) from dl_exc

    def _count_hype_words(self, text: str) -> int:
        """
        Count occurrences of hype words and phrases in the text (case-insensitive).
        Longer phrases are matched first to prevent partial double-counting.

        Args:
            text: Input string.

        Returns:
            int: Number of hype term matches found.
        """
        if not text:
            return 0

        lower_text = text.lower()
        matched_spans: List[tuple[int, int]] = []
        count = 0

        for term in self.hype_words:
            term_clean = term.strip().lower()
            if not term_clean:
                continue

            pattern = r"\b" + re.escape(term_clean) + r"\b"
            for match in re.finditer(pattern, lower_text):
                span = match.span()
                # Check for overlap with already matched phrases
                if not any(max(span[0], m[0]) < min(span[1], m[1]) for m in matched_spans):
                    matched_spans.append(span)
                    count += 1

        return count

    def _compute_hype_score(
        self,
        text: str,
        compound: float,
        booster_mode: str = "formula",
    ) -> float:
        """
        Compute a normalized hype score clamped to [0.0, 1.0].

        Formula: Base = abs(compound) + caps_ratio*0.2 + exclamation_density*0.1 + hype_word_count*0.1
        Normalized to 0-1.

        Args:
            text: Segment text.
            compound: VADER compound polarity score (-1.0 to 1.0).
            booster_mode: 'formula' (default, uses caps_ratio*0.2 and exclamation_density*0.1)
                          or 'discrete' (uses +0.1 per ALL CAPS word, +0.05 per exclamation mark).

        Returns:
            float: Normalized hype score between 0.0 and 1.0.
        """
        base = abs(float(compound))
        if not text or not str(text).strip():
            return max(0.0, min(1.0, round(base, 4)))

        text_str = str(text)

        # Extract words for ratio and caps analysis
        words = [w for w in re.findall(r"\b[A-Za-z0-9_']+\b", text_str) if any(c.isalnum() for c in w)]
        total_words = len(words)

        # ALL CAPS words: uppercase words with alphabetic chars (ignore standalone 'I' and 'A')
        caps_words = [
            w for w in words
            if w.isupper() and any(c.isalpha() for c in w) and (len(w) > 1 or w not in ("I", "A"))
        ]
        caps_count = len(caps_words)
        caps_ratio = (caps_count / total_words) if total_words > 0 else 0.0

        # Exclamation marks
        exclamation_count = text_str.count("!")
        exclamation_density = (
            (exclamation_count / total_words) if total_words > 0
            else (1.0 if exclamation_count > 0 else 0.0)
        )

        # Hype words / phrases count
        hype_word_count = self._count_hype_words(text_str)

        if booster_mode == "discrete":
            # Discrete boosters: ALL CAPS (+0.1 each), exclamation (+0.05 each), hype words (+0.1 each)
            raw_score = (
                base
                + (caps_count * 0.1)
                + (exclamation_count * 0.05)
                + (hype_word_count * 0.1)
            )
        else:
            # Standard formula: Base = abs(compound) + caps_ratio*0.2 + exclamation_density*0.1 + hype_word_count*0.1
            raw_score = (
                base
                + (caps_ratio * 0.2)
                + (exclamation_density * 0.1)
                + (hype_word_count * 0.1)
            )

        # Normalize / Clamp to [0, 1]
        clamped_score = max(0.0, min(1.0, float(raw_score)))
        return round(clamped_score, 4)

    def _get_hype_level(self, score: float) -> str:
        """
        Map a normalized hype score in [0.0, 1.0] to a categorical hype level:
        'low', 'medium', 'high', or 'extreme'.

        Args:
            score: Normalized score between 0.0 and 1.0.

        Returns:
            str: One of 'low', 'medium', 'high', 'extreme'.
        """
        if score >= 0.8:
            return "extreme"
        elif score >= 0.6:
            return "high"
        elif score >= 0.35:
            return "medium"
        else:
            return "low"

    def analyze_segments(
        self,
        segments: List[Dict[str, Any]],
        booster_mode: str = "formula",
    ) -> List[Dict[str, Any]]:
        """
        Analyze a list of transcribed segments.
        For each segment:
            - Run VADER to get the compound score.
            - Apply hype boosters (ALL CAPS, exclamation marks, hype words).
            - Clamp score to [0, 1].
            - Add 'sentiment_score' and 'hype_level' (low/medium/high/extreme) to each segment.

        Args:
            segments: List of segment dictionaries (each containing at least 'text').
            booster_mode: Mode for hype boosters ('formula' or 'discrete').

        Returns:
            List[Dict[str, Any]]: The analyzed segments with sentiment_score and hype_level added.
        """
        if not segments:
            return []

        for seg in segments:
            if not isinstance(seg, dict):
                continue

            text = seg.get("text", "")
            if text:
                try:
                    polarity = self.vader.polarity_scores(text)
                    compound = float(polarity.get("compound", 0.0))
                except Exception as exc:
                    logger.warning("VADER polarity scoring failed for text '%s': %s", text, exc)
                    compound = 0.0
            else:
                compound = 0.0

            # Compute normalized hype score clamped to [0, 1]
            score = self._compute_hype_score(text, compound, booster_mode=booster_mode)
            hype_level = self._get_hype_level(score)

            # Add to segment
            seg["sentiment_score"] = score
            seg["hype_level"] = hype_level

        return segments

    def analyze(self, transcript_data: Any) -> Dict[str, Any]:
        """
        Pipeline-compatible analysis method.
        Accepts transcript_data (dict with 'segments' or list of segments),
        runs analyze_segments, and returns timestamps and scores.

        Args:
            transcript_data: Dictionary containing 'segments' or list of segment dicts.

        Returns:
            Dict containing 'segments', 'timestamps', 'sentiment_scores', and 'scores'.
        """
        if isinstance(transcript_data, dict):
            segments = transcript_data.get("segments", [])
        elif isinstance(transcript_data, list):
            segments = transcript_data
        else:
            segments = []

        analyzed_segments = self.analyze_segments(segments)
        timestamps = []
        scores = []
        for s in analyzed_segments:
            start = float(s.get("start", 0.0))
            end = float(s.get("end", start + 2.0))
            timestamps.append(round((start + end) / 2.0, 3))
            scores.append(float(s.get("sentiment_score", 0.0)))

        return {
            "segments": analyzed_segments,
            "timestamps": timestamps,
            "sentiment_scores": scores,
            "scores": scores,
        }

    def get_score_at_time(
        self,
        analyzed_segments: List[Dict[str, Any]],
        timestamp: Union[float, int],
        default: float = 0.0,
    ) -> float:
        """
        Find the segment containing the given timestamp and return its score.

        Args:
            analyzed_segments: List of segment dicts with 'start', 'end', and 'sentiment_score'.
            timestamp: Time in seconds.
            default: Fallback score if no segment covers the timestamp (default 0.0).

        Returns:
            float: The sentiment_score of the containing segment, or default if not found.
        """
        if not analyzed_segments:
            return default

        ts = float(timestamp)
        for seg in analyzed_segments:
            if not isinstance(seg, dict):
                continue

            try:
                seg_start = float(seg.get("start", 0.0))
                seg_end = float(seg.get("end", 0.0))
            except (ValueError, TypeError):
                continue

            if seg_start <= ts <= seg_end:
                score = seg.get("sentiment_score")
                if score is None:
                    score = seg.get("hype_score", default)
                try:
                    return float(score)
                except (ValueError, TypeError):
                    return default

        return default
