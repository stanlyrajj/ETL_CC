"""
technical.py — Technical analysis API endpoints.

POST   /api/technical/{paper_id}/analyze      — start analysis or return cached result
GET    /api/technical/{paper_id}/cached        — check what sections are already cached
GET    /api/technical/{queue_key}/progress     — SSE stream for live analysis
DELETE /api/technical/{paper_id}/cache         — bust cache so next call regenerates

Cache strategy
--------------
Each section is cached independently under:
  cache_type = "technical_section"
  cache_key  = "{paper_id}::{section_key}"

On POST /analyze:
  - If ALL sections exist in cache, return them immediately as a flat list
    (no SSE needed — frontend can render instantly).
  - If ANY section is missing, run full analysis for all sections, overwriting
    any partials (simpler than partial re-runs; technical analysis is fast enough).

The response shape tells the frontend which path was taken:
  { cached: true,  sections: [...] }   — instant, no SSE
  { cached: false, queue_key: "..." }  — SSE path, same as before
"""

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import delete as sql_delete

from database import db, GeneratedCache
from content.technical import TECHNICAL_SECTIONS, generate_section
from api.progress import get_or_create_generate_queue, push_generate_event

logger = logging.getLogger(__name__)

router = APIRouter()

_CACHE_TYPE = "technical_section"


def _section_cache_key(paper_id: str, section_key: str) -> str:
    return f"{paper_id}::{section_key}"


async def _load_all_from_cache(paper_id: str) -> list[dict] | None:
    """Return all sections from cache if every section is present, else None."""
    sections = []
    async with db.session() as sess:
        for section in TECHNICAL_SECTIONS:
            ck = _section_cache_key(paper_id, section["key"])
            cached = await db.get_cache(sess, _CACHE_TYPE, ck)
            if cached is None:
                return None  # incomplete — need to regenerate
            sections.append({
                "section_key":   section["key"],
                "section_label": section["label"],
                "content":       cached.content,
            })
    return sections


async def _run_technical_analysis(paper_id: str, queue_key: str) -> None:
    """
    Generate all technical sections sequentially.
    Caches each section to SQLite as it completes.
    Pushes SSE events for live progress tracking.
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
            prev_sections += f"\n\n## {section['label']}\n{content[:800]}…"

            # Persist to cache
            async with db.session() as sess:
                await db.set_cache(
                    sess,
                    paper_id=paper_id,
                    cache_type=_CACHE_TYPE,
                    cache_key=_section_cache_key(paper_id, section["key"]),
                    content=content,
                )

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

    await push_generate_event(queue_key, "done", {
        "queue_key": queue_key,
        "success":   True,
        "paper_id":  paper_id,
    })
    logger.info("Technical analysis complete: paper=%s", paper_id)


@router.post("/technical/{paper_id}/analyze")
async def start_technical_analysis(paper_id: str):
    """
    Start (or skip) technical analysis.

    Returns one of two shapes:
      { cached: true,  sections: [{section_key, section_label, content}, ...] }
      { cached: false, queue_key: "...", paper_id: "...", sections: [...defs...] }
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

    # Fast path — all sections already cached
    cached_sections = await _load_all_from_cache(paper_id)
    if cached_sections is not None:
        logger.info("Technical analysis cache hit: paper=%s", paper_id)
        return {
            "cached":   True,
            "paper_id": paper_id,
            "sections": cached_sections,
        }

    # Slow path — generate via SSE
    queue_key = f"tech_{paper_id}_{uuid.uuid4().hex[:8]}"
    get_or_create_generate_queue(queue_key)
    asyncio.create_task(_run_technical_analysis(paper_id, queue_key))

    return {
        "cached":    False,
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


@router.delete("/technical/{paper_id}/cache")
async def bust_technical_cache(paper_id: str):
    """
    Delete all cached technical sections for a paper.
    The next POST /analyze will regenerate from scratch.
    """
    async with db.session() as sess:
        paper = await db.get_paper(sess, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found.")

    async with db.session() as sess:
        result = await sess.execute(
            sql_delete(GeneratedCache).where(
                GeneratedCache.paper_id == paper_id,
                GeneratedCache.cache_type == _CACHE_TYPE,
            )
        )
        deleted = result.rowcount

    logger.info("Busted technical cache for paper=%s (%d entries)", paper_id, deleted)
    return {"success": True, "paper_id": paper_id, "deleted": deleted}