import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from backend.app.core.progress import progress_tracker
from backend.app.logging import logger

router = APIRouter(tags=["websocket"])


@router.websocket("/jobs/{job_id}/events")
@router.websocket("/ws/jobs/{job_id}")
async def job_events_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    queue = progress_tracker.subscribe(job_id)

    try:
        while True:
            try:
                event_data = await asyncio.wait_for(queue.get(), timeout=20.0)
                await websocket.send_json(event_data)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping", "job_id": job_id})
    except (WebSocketDisconnect, ConnectionResetError):
        logger.info("websocket_disconnected", job_id=job_id)
    finally:
        progress_tracker.unsubscribe(job_id, queue)


@router.get("/jobs/{job_id}/events")
async def job_events_sse(job_id: str):
    """Server-Sent Events fallback for HTTP clients and environments without WebSockets."""
    async def event_generator():
        queue = progress_tracker.subscribe(job_id)
        try:
            while True:
                try:
                    event_data = await asyncio.wait_for(queue.get(), timeout=20.0)
                    yield f"data: {json.dumps(event_data)}\n\n"
                    if event_data.get("stage") in ("done", "failed"):
                        break
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            progress_tracker.unsubscribe(job_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
