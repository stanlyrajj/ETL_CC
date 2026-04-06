"""
technical.py — Technical analysis API endpoints.

POST /api/technical/{paper_id}/analyze      — start analysis, returns queue_key
GET  /api/technical/{queue_key}/progress    — SSE stream, pushes sections as completed
"""

import asyncio
import logging
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from database import db
from content.technical import TECHNICAL_SECTIONS, generate_section
from api.progress import get_or_create_generate_queue, push_generate_event

logger = logging.getLogger(__name__)

router = APIRouter()


async def _run_technical_analysis(paper_id: str, queue_key: str) -> None:
    """
    Generate all five technical sections sequentially.
    Pushes SSE events per section as they complete.
    Never raises — all failures are caught and pushed as events.
    """
    prev_sections = ""

    await push_generate_event(queue_key, "started", {
        "queue_key": queue_key,
        "paper_id":  paper_id,
        "total":     len(TECHNICAL_SECTIONS),
    })

    for i, section in enumerate(TECHNICAL_SECTIONS):
        try:
            content = await generate_section(
                paper_id=paper_id,
                section_key=section["key"],
                section_label=section["label"],
                prev_sections=prev_sections,
            )
            # Accumulate for continuity context in subsequent sections.
            # Keep only a summary to avoid exceeding context limits — use
            # the first 800 chars of each section as a brief prior.
            prev_sections += f"\n\n## {section['label']}\n{content[:800]}…"

            await push_generate_event(queue_key, "section", {
                "queue_key":     queue_key,
                "section_index": i,
                "section_key":   section["key"],
                "section_label": section["label"],
                "content":       content,
                "total":         len(TECHNICAL_SECTIONS),
            })

        except Exception as exc:
            error = str(exc)
            logger.error("Technical section failed: paper=%s section=%s: %s",
                         paper_id, section["key"], error)
            await push_generate_event(queue_key, "section_failed", {
                "queue_key":     queue_key,
                "section_index": i,
                "section_key":   section["key"],
                "section_label": section["label"],
                "error":         error,
            })
            # Continue with remaining sections even if one fails

    await push_generate_event(queue_key, "done", {
        "queue_key": queue_key,
        "success":   True,
        "paper_id":  paper_id,
    })
    logger.info("Technical analysis complete: paper=%s", paper_id)


@router.post("/technical/{paper_id}/analyze")
async def start_technical_analysis(paper_id: str):
    """
    Start technical analysis as a background task.
    Returns a queue_key to track progress via SSE.
    """
    async with db.session() as sess:
        paper = await db.get_paper(sess, paper_id)

    if paper is None:
        raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id!r}")
    if paper.pipeline_stage != "processed":
        raise HTTPException(
            status_code=409,
            detail=f"Paper is not ready yet. Current stage: {paper.pipeline_stage!r}.",
        )

    queue_key = f"tech_{paper_id}_{uuid.uuid4().hex[:8]}"
    get_or_create_generate_queue(queue_key)

    asyncio.create_task(_run_technical_analysis(paper_id, queue_key))

    return {
        "queue_key": queue_key,
        "paper_id":  paper_id,
        "sections":  TECHNICAL_SECTIONS,
        "message":   "Technical analysis started. Track progress via SSE.",
    }


@router.get("/technical/{queue_key}/progress")
async def technical_progress(queue_key: str):
    """SSE stream for technical analysis progress."""
    from api.progress import _generate_stream
    return StreamingResponse(
        _generate_stream(queue_key),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
