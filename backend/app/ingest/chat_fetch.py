from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import json
from pathlib import Path
from typing import List
from backend.app.models import ChatMsg
from backend.app.logging import logger

try:
    from chat_downloader import ChatDownloader
except ImportError:
    ChatDownloader = None


def fetch_chat(source_url: str | None, out_json: Path | None = None) -> List[ChatMsg]:
    if not source_url:
        return []

    if ChatDownloader is None:
        logger.warning("chat_downloader_not_installed", fallback="no_chat")
        return []

    messages: List[ChatMsg] = []

    def _worker():
        try:
            downloader = ChatDownloader()
            # Request chat stream with message ceiling to avoid infinite hangs on huge streams
            chat_stream = downloader.get_chat(source_url, max_messages=5000)
            for item in chat_stream:
                time_in_sec = float(item.get("time_in_seconds", 0.0))
                author = item.get("author", {}).get("name", "unknown")
                text = item.get("message", "")
                emotes = [e.get("name", "") for e in item.get("emoticons", []) if isinstance(e, dict)]
                if text:
                    messages.append(
                        ChatMsg(
                            t=time_in_sec,
                            author=author,
                            text=text,
                            emotes=emotes,
                        )
                    )
        except Exception as e:
            logger.warning("chat_fetch_failed", url=source_url, error=str(e), fallback="empty_chat")
        return messages

    # Strict 6-second timeout: never stall video ingestion if chat replay is unavailable
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            fut = executor.submit(_worker)
            messages = fut.result(timeout=6.0)
    except (FutureTimeout, Exception) as te:
        logger.warning("chat_fetch_timeout_or_error", url=source_url, error=str(te), fallback="empty_chat")
        messages = []

    if out_json:
        try:
            out_json.parent.mkdir(parents=True, exist_ok=True)
            serializable = [m.model_dump() for m in messages]
            out_json.write_text(json.dumps(serializable, indent=2))
        except Exception as e:
            logger.warning("failed_to_cache_chat_json", error=str(e))

    return messages
