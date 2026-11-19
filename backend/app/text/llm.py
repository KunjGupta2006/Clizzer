from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path
import random
import re
import time
from typing import Any, Dict, Optional, Type
import httpx
from pydantic import BaseModel
from backend.app.config import settings
from backend.app.errors import LLMUnavailable
from backend.app.logging import logger


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class LLMRateLimiter:
    """Thread-safe rate limiter based on requests-per-minute (RPM)."""
    def __init__(self, rpm: int = 20):
        self.rpm = max(1, rpm)
        self.interval = 60.0 / self.rpm
        self.last_call_time = 0.0

    def wait(self) -> None:
        now = time.time()
        elapsed = now - self.last_call_time
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call_time = time.time()


_GLOBAL_RATE_LIMITER = LLMRateLimiter(rpm=settings.GROQ_RPM)


class LLMProvider(ABC):
    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """Complete a system/user prompt returning parsed dictionary matching schema."""
        pass


class PersistentCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_key(self, model: str, system: str, user: str) -> str:
        payload = f"{model}::{system}::{user}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, model: str, system: str, user: str) -> Optional[Dict[str, Any]]:
        key = self._get_key(model, system, user)
        cache_file = self.cache_dir / f"llm_{key}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def set(self, model: str, system: str, user: str, data: Dict[str, Any]) -> None:
        key = self._get_key(model, system, user)
        cache_file = self.cache_dir / f"llm_{key}.json"
        try:
            cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("cache_write_failed", error=str(e))


class GroqProvider(LLMProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        client: Optional[httpx.Client] = None,
    ):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self.fallback_model = fallback_model or settings.GROQ_FALLBACK_MODEL
        self.client = client
        self.cache = PersistentCache(settings.CACHE_DIR)
        self.rate_limiter = LLMRateLimiter(rpm=10000) if client is not None else _GLOBAL_RATE_LIMITER

    def complete(
        self,
        system: str,
        user: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise LLMUnavailable("GROQ_API_KEY not set")

        # Check persistent cache
        cached = self.cache.get(self.model, system, user)
        if cached is not None:
            logger.debug("llm_cache_hit", model=self.model)
            return cached

        active_model = self.model
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        retries = 3
        for attempt in range(retries):
            self.rate_limiter.wait()
            payload = {
                "model": active_model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
            }

            try:
                cli = self.client or httpx.Client(timeout=30.0)
                try:
                    resp = cli.post(url, headers=headers, json=payload)
                finally:
                    if self.client is None:
                        cli.close()

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_time = float(retry_after) if retry_after else (2.0 ** attempt + random.uniform(0.1, 0.5))
                    logger.warning("groq_429_rate_limit", retry_after=sleep_time, attempt=attempt)
                    time.sleep(sleep_time)
                    continue

                if resp.status_code >= 500:
                    sleep_time = 2.0 ** attempt + random.uniform(0.1, 0.5)
                    logger.warning("groq_server_error", status_code=resp.status_code, attempt=attempt)
                    time.sleep(sleep_time)
                    continue

                if resp.status_code == 404 or resp.status_code == 400:
                    # Model might be rejected, try fallback model
                    if active_model != self.fallback_model:
                        logger.warning("groq_model_rejected_falling_back", original=active_model, fallback=self.fallback_model)
                        active_model = self.fallback_model
                        continue

                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                raw_json = strip_code_fences(content)
                parsed = json.loads(raw_json)

                if schema is not None:
                    # Validate schema
                    try:
                        validated = schema.model_validate(parsed)
                        parsed_dict = validated.model_dump()
                    except Exception as val_err:
                        # Append validation error message and retry
                        logger.warning("schema_validation_error", error=str(val_err), attempt=attempt)
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": f"The response was invalid: {val_err}. Please output valid JSON matching the exact schema."
                        })
                        continue
                else:
                    parsed_dict = parsed

                # Cache successful response
                self.cache.set(self.model, system, user, parsed_dict)
                return parsed_dict

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("json_parse_error", error=str(e), attempt=attempt)
                messages.append({"role": "user", "content": f"Failed to parse JSON: {e}. Output valid JSON only."})
            except httpx.HTTPError as e:
                logger.error("groq_http_error", error=str(e), attempt=attempt)
                if attempt == retries - 1:
                    raise LLMUnavailable(f"Groq API HTTP error: {e}") from e

        raise LLMUnavailable(f"Groq completions failed after {retries} retries")


class OllamaProvider(LLMProvider):
    def __init__(self, host: Optional[str] = None, model: Optional[str] = None):
        self.host = host or settings.OLLAMA_HOST
        self.model = model or settings.OLLAMA_MODEL

    def complete(
        self,
        system: str,
        user: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.host.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": f"{system}\n\nUser: {user}",
            "format": "json",
            "stream": False,
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                response_text = resp.json().get("response", "{}")
                parsed = json.loads(strip_code_fences(response_text))
                if schema is not None:
                    return schema.model_validate(parsed).model_dump()
                return parsed
        except Exception as e:
            raise LLMUnavailable(f"Ollama completion failed: {e}") from e


class MockProvider(LLMProvider):
    def __init__(
        self,
        mock_response: Optional[Dict[str, Any]] = None,
        simulate_malformed_count: int = 0,
        simulate_429: bool = False,
    ):
        self.mock_response = mock_response or {}
        self.simulate_malformed_count = simulate_malformed_count
        self.simulate_429 = simulate_429
        self.calls_made = 0
        self.last_retry_after: Optional[float] = None

    def complete(
        self,
        system: str,
        user: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        self.calls_made += 1
        if self.simulate_429:
            self.last_retry_after = 0.05
            time.sleep(self.last_retry_after)
            # clear flag after one trigger
            self.simulate_429 = False

        if self.simulate_malformed_count > 0:
            self.simulate_malformed_count -= 1
            raise LLMUnavailable("Malformed JSON returned by provider")

        if schema is not None and self.mock_response:
            return schema.model_validate(self.mock_response).model_dump()
        return self.mock_response
