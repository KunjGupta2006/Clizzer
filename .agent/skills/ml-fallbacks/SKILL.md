---
name: ml-fallbacks
description: Architectural design for graceful model fallbacks across LLM, Diarization, Vision, and Audio.
---
# ML Fallbacks

1. Every model loader returns `object | None`, logs a structured warning, and never raises unhandled exceptions.
2. Each stage output contains `available: bool` and `fallback_reason: str | None`.
3. Fallback chains:
   - LLM: Groq API -> Ollama (if reachable) -> Heuristic text_score.
   - Diarization: PyAnnote 3.x (with HF_TOKEN) -> SpeechBrain ECAPA / clustering -> Single speaker.
   - GPU: CUDA float16 -> CPU int8.
   - Faces: InsightFace -> MediaPipe -> No face series (weight 0).
4. Missing modality: produce zero series, set weight to 0, and renormalize remaining weights to 1.0.
5. Singleton model cache; free GPU memory between stages.
6. Test both primary (mocked) and fallback paths.
