import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.app.core.progress import progress_tracker
from backend.app.logging import logger

router = APIRouter(tags=["websocket"])


@router.websocket("/jobs/{job_id}/events")
async def job_events_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    queue = progress_tracker.subscribe(job_id)

    try:
        while True:
            # Wait for next event with a timeout for ping/keepalive
            try:
                event_data = await asyncio.wait_for(queue.get(), timeout=20.0)
                await websocket.send_json(event_data)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping", "job_id": job_id})
    except (WebSocketDisconnect, ConnectionResetError):
        logger.info("websocket_disconnected", job_id=job_id)
    finally:
        progress_tracker.unsubscribe(job_id, queue)
