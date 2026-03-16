"""
progress.py — Server-Sent Events (SSE) for pipeline and generation progress.

Two queues are maintained as module-level dicts:
  _paper_queues   — keyed by paper_id
  _generate_queues — keyed by queue_key

Any part of the pipeline pushes events by calling the push_* helpers.
The SSE endpoints drain their queue and stream events to the client.
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# ── In-memory event queues ─────────────────────────────────────────────────────
# Each key maps to an asyncio.Queue of event dicts.
# Queues are created on first push and removed when the client disconnects.

_paper_queues:    dict[str, asyncio.Queue] = {}
_generate_queues: dict[str, asyncio.Queue] = {}


# ── Queue helpers (called by pipeline and generation tasks) ───────────────────

def get_or_create_paper_queue(paper_id: str) -> asyncio.Queue:
    if paper_id not in _paper_queues:
        _paper_queues[paper_id] = asyncio.Queue()
    return _paper_queues[paper_id]


def get_or_create_generate_queue(queue_key: str) -> asyncio.Queue:
    if queue_key not in _generate_queues:
        _generate_queues[queue_key] = asyncio.Queue()
    return _generate_queues[queue_key]


async def push_paper_event(paper_id: str, event: str, data: dict) -> None:
    """Push a progress event for a paper pipeline."""
    q = get_or_create_paper_queue(paper_id)
    await q.put({"event": event, "data": data})


async def push_generate_event(queue_key: str, event: str, data: dict) -> None:
    """Push a progress event for a content generation task."""
    q = get_or_create_generate_queue(queue_key)
    await q.put({"event": event, "data": data})


# ── SSE formatting ────────────────────────────────────────────────────────────

def _format_sse(event: str, data: dict) -> str:
    """Format a single SSE message."""
    payload = json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


# ── SSE stream generators ─────────────────────────────────────────────────────

async def _paper_stream(paper_id: str) -> AsyncGenerator[str, None]:
    """Yield SSE events for a paper's pipeline progress."""
    q = get_or_create_paper_queue(paper_id)
    last_heartbeat = time.monotonic()

    try:
        while True:
            # Heartbeat every 30 seconds to keep the connection alive
            now = time.monotonic()
            if now - last_heartbeat >= 30:
                yield _format_sse("heartbeat", {"paper_id": paper_id, "ts": int(now)})
                last_heartbeat = now

            try:
                item = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            yield _format_sse(item["event"], item["data"])

            # Stop streaming after terminal events
            if item["event"] == "done":
                break

    except asyncio.CancelledError:
        logger.debug("SSE paper stream cancelled: paper_id=%s", paper_id)
    finally:
        # Clean up queue when client disconnects
        _paper_queues.pop(paper_id, None)


async def _generate_stream(queue_key: str) -> AsyncGenerator[str, None]:
    """Yield SSE events for a content generation task."""
    q = get_or_create_generate_queue(queue_key)
    last_heartbeat = time.monotonic()

    try:
        while True:
            now = time.monotonic()
            if now - last_heartbeat >= 30:
                yield _format_sse("heartbeat", {"queue_key": queue_key, "ts": int(now)})
                last_heartbeat = now

            try:
                item = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            yield _format_sse(item["event"], item["data"])

            if item["event"] in ("done", "failed"):
                break

    except asyncio.CancelledError:
        logger.debug("SSE generate stream cancelled: queue_key=%s", queue_key)
    finally:
        _generate_queues.pop(queue_key, None)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/papers/{paper_id}/progress")
async def paper_progress(paper_id: str):
    """
    SSE stream for pipeline progress on a paper.
    Events: progress (stage updates), done (success/failure), heartbeat.
    """
    return StreamingResponse(
        _paper_stream(paper_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/generate/{queue_key}/progress")
async def generate_progress(queue_key: str):
    """
    SSE stream for content generation progress.
    Events: started, completed, failed, done.
    """
    return StreamingResponse(
        _generate_stream(queue_key),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
