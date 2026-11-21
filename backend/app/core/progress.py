import asyncio
import json
from typing import Any, Dict, List, Set
from backend.app.db import engine
from sqlmodel import Session, select
from backend.app.models import Job
from backend.app.logging import logger


class ProgressTracker:
    def __init__(self):
        # job_id -> set of asyncio.Queue
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}

    def subscribe(self, job_id: str) -> asyncio.Queue:
        if job_id not in self._subscribers:
            self._subscribers[job_id] = set()
        queue = asyncio.Queue()
        self._subscribers[job_id].add(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        if job_id in self._subscribers:
            self._subscribers[job_id].discard(queue)
            if not self._subscribers[job_id]:
                del self._subscribers[job_id]

    async def broadcast(self, job_id: str, data: Dict[str, Any]) -> None:
        if job_id in self._subscribers:
            for q in list(self._subscribers[job_id]):
                await q.put(data)

    def update_job_sync(
        self,
        job_id: str,
        stage: str,
        progress: float,
        status: str = "running",
        timings: Dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with Session(engine) as session:
            job = session.exec(select(Job).where(Job.id == job_id)).first()
            if job:
                # Terminal status guarantee: do not regress once done or failed
                if job.status in ("done", "failed") and status not in ("done", "failed"):
                    return
                job.stage = stage
                job.progress = float(progress)
                if status != "running":
                    job.status = status
                if timings:
                    t = job.timings
                    t.update(timings)
                    job.timings = t
                if error:
                    job.error = error
                session.add(job)
                session.commit()

        # Try scheduling broadcast on event loop if one is active
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                msg = {
                    "job_id": job_id,
                    "stage": stage,
                    "progress": progress,
                    "status": status,
                    "timings": timings or {},
                    "error": error,
                }
                asyncio.run_coroutine_threadsafe(self.broadcast(job_id, msg), loop)
        except Exception:
            pass


progress_tracker = ProgressTracker()
