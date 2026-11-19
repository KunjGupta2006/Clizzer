import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional
from backend.app.config import settings
from backend.app.logging import logger


class PipelineCache:
    def __init__(self, cache_dir: Path = settings.CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_job_cache_dir(self, input_key: str) -> Path:
        key_hash = hashlib.sha256(input_key.encode("utf-8")).hexdigest()
        job_dir = self.cache_dir / key_hash
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def load_stage(self, input_key: str, stage_name: str) -> Optional[Dict[str, Any]]:
        target_file = self.get_job_cache_dir(input_key) / f"{stage_name}.json"
        if target_file.exists():
            try:
                data = json.loads(target_file.read_text(encoding="utf-8"))
                logger.info("stage_cache_hit", stage=stage_name, key=input_key)
                return data
            except Exception as e:
                logger.warning("stage_cache_read_error", error=str(e))
                return None
        return None

    def save_stage(self, input_key: str, stage_name: str, data: Dict[str, Any]) -> None:
        target_file = self.get_job_cache_dir(input_key) / f"{stage_name}.json"
        try:
            target_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("stage_cache_write_error", stage=stage_name, error=str(e))


pipeline_cache = PipelineCache()
