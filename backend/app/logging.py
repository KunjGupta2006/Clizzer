import logging
import sys
from typing import Any, MutableMapping
import structlog

# Secret keys to redact
SENSITIVE_KEYS = {"api_key", "groq_api_key", "hf_token", "gemini_api_key", "password", "token", "secret"}


def mask_secrets(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    for key, val in list(event_dict.items()):
        if any(secret_term in key.lower() for secret_term in SENSITIVE_KEYS):
            if isinstance(val, str) and val:
                event_dict[key] = f"***redacted(len={len(val)})***"
    return event_dict


def setup_logging(log_level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            mask_secrets,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


setup_logging()
logger = structlog.get_logger("clizzer")
