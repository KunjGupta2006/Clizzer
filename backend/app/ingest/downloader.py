import json
from pathlib import Path
import shutil
from typing import Any, Dict, Tuple
from backend.app.errors import SourceUnavailable
from backend.app.ingest.ffmpeg_utils import validate_source
from backend.app.logging import logger

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


def download_video(
    job_id: str,
    source_url: str | None,
    file_path: str | None,
    jobs_dir: Path,
) -> Tuple[Path, Dict[str, Any]]:
    target_dir = jobs_dir / job_id
    target_dir.mkdir(parents=True, exist_ok=True)
    source_mp4 = target_dir / "source.mp4"
    metadata_json = target_dir / "metadata.json"

    metadata: Dict[str, Any] = {
        "job_id": job_id,
        "source_url": source_url,
        "file_path": file_path,
        "title": "Uploaded Video",
        "duration": 0.0,
    }

    if file_path:
        local_src = Path(file_path)
        if not local_src.exists():
            raise SourceUnavailable(f"Local file does not exist: {file_path}")
        try:
            shutil.copy2(local_src, source_mp4)
            info = validate_source(source_mp4)
            duration = float(info.get("format", {}).get("duration", 0.0))
            metadata["duration"] = duration
            metadata["title"] = local_src.stem
            metadata_json.write_text(json.dumps(metadata, indent=2))
            return source_mp4, metadata
        except Exception as e:
            raise SourceUnavailable(f"Failed to ingest local file {file_path}: {e}") from e

    if not source_url:
        raise SourceUnavailable("Neither source_url nor file_path provided")

    if yt_dlp is None:
        raise SourceUnavailable("yt-dlp is not installed")

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(target_dir / "source.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "live_from_start": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "web"]
            }
        },
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            extracted_info = ydl.extract_info(source_url, download=True)
            if not extracted_info:
                raise SourceUnavailable(f"Could not extract information from {source_url}")

            metadata["title"] = extracted_info.get("title", "Video")
            metadata["duration"] = float(extracted_info.get("duration", 0.0))
            metadata["author"] = extracted_info.get("uploader", "")
            metadata["channel"] = extracted_info.get("channel", "")

            # Look for source.mp4 or any merged output
            if not source_mp4.exists():
                potential = list(target_dir.glob("source.*"))
                if potential:
                    # Rename first match to source.mp4
                    potential[0].rename(source_mp4)

            validate_source(source_mp4)
            metadata_json.write_text(json.dumps(metadata, indent=2))
            return source_mp4, metadata

    except Exception as e:
        logger.error("ytdlp_download_failed", url=source_url, error=str(e))
        raise SourceUnavailable(f"Failed to download video from {source_url}: {e}") from e
