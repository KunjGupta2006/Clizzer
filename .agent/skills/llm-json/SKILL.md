---
name: llm-json
description: Protocol for reliable structured JSON generation from LLMs with retries and rate limiting.
---
# LLM JSON Protocol

1. Single interface: `LLMProvider.complete(system: str, user: str, schema: Type[BaseModel] | None) -> dict`.
2. GroqProvider:
   - Endpoint: `https://api.groq.com/openai/v1/chat/completions`
   - Header: `Authorization: Bearer GROQ_API_KEY`
   - Model: `env GROQ_MODEL` (default: `"llama-3.3-70b-versatile"`; on rejection, fall back to `"llama-3.1-8b-instant"`)
   - `response_format: {"type": "json_object"}`, `temperature: 0`
3. Free-tier protection:
   - Batch max 5 windows per call.
   - Cap prompt size around ~3k tokens.
   - Token bucket / rate limiter (`GROQ_RPM`, default 20).
   - Honor `Retry-After` header on HTTP 429.
   - Exponential backoff with jitter on 429 and 5xx.
   - Persistent cache keyed on `sha256(model + prompt + payload)` in `data/cache/`.
4. Validation:
   - Validate response with Pydantic.
   - Strip markdown code fences (```json ... ```) before parsing.
   - Retry up to 3 times appending the validation error message.
   - Clamp scores to 0-10 range.
   - On retry exhaustion, raise `LLMUnavailable` so caller falls back to heuristic text scoring.
5. OllamaProvider optional (`OLLAMA_HOST`, `OLLAMA_MODEL`, `format="json"`).
6. Tests must use `MockProvider` (including malformed JSON and 429 simulations); never call real network in tests.
