"""
study.py — Study assistant API endpoints.

POST   /api/study/{paper_id}/outline    — return cached outline or generate + cache
POST   /api/study/{paper_id}/section    — return cached section or generate + cache
POST   /api/study/{paper_id}/flashcards — return cached flashcards or generate + cache
GET    /api/study/{paper_id}/cache      — return full cache status for a paper
DELETE /api/study/{paper_id}/cache      — bust all study cache so next call regenerates

Cache strategy
--------------
All generated content is stored in the generated_cache table keyed by:
  outline    → cache_type="study_outline",    cache_key=paper_id
  section    → cache_type="study_section",    cache_key="{paper_id}::{section_title}"
  flashcards → cache_type="study_flashcards", cache_key=paper_id

Race safety
-----------
Each endpoint opens exactly ONE session that covers both the cache read and
the cache write.  set_cache uses a SQLite INSERT … ON CONFLICT DO UPDATE so
the write itself is a single atomic statement — no TOCTOU gap.
"""

import json
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from database import db, GeneratedCache
from content.study import generate_outline, generate_section, generate_flashcards

logger = logging.getLogger(__name__)

router = APIRouter()


class SectionRequest(BaseModel):
    section_title:       str = Field(..., min_length=1)
    section_description: str = Field("")
    level:               str = Field("beginner")
    section_index:       int = Field(0, ge=0)  # position in outline — used for ordering on restore


def _check_paper(paper):
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    if paper.pipeline_stage != "processed":
        raise HTTPException(
            status_code=409,
            detail=f"Paper is not ready yet. Current stage: {paper.pipeline_stage!r}.",
        )


@router.post("/study/{paper_id}/outline")
async def study_outline(paper_id: str):
    """
    Return the study outline for a paper.
    Serves from cache instantly if available; generates and caches on first call.
    One session covers both the cache read and the cache write.
    """
    async with db.session() as sess:
        paper = await db.get_paper(sess, paper_id)
        _check_paper(paper)
        cached = await db.get_cache(sess, "study_outline", paper_id)

    if cached is not None:
        try:
            outline = json.loads(cached.content)
            return {"paper_id": paper_id, "outline": outline, "cached": True}
        except (json.JSONDecodeError, ValueError):
            pass  # corrupt cache — fall through to regenerate

    # Generate (outside the session — this is the slow LLM call)
    try:
        result = await generate_outline(paper_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Study outline failed for %s: %s", paper_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to generate outline: {exc}")

    # Atomic upsert in a single session
    async with db.session() as sess:
        await db.set_cache(
            sess,
            paper_id=paper_id,
            cache_type="study_outline",
            cache_key=paper_id,
            content=json.dumps(result),
        )

    return {"paper_id": paper_id, "outline": result, "cached": False}


@router.post("/study/{paper_id}/section")
async def study_section(paper_id: str, request: SectionRequest):
    """
    Return the teaching content for one section.
    Serves from cache instantly if available; generates and caches on first call.
    One session covers both the cache read and the cache write.
    """
    cache_key = f"{paper_id}::{request.section_title}"

    async with db.session() as sess:
        paper = await db.get_paper(sess, paper_id)
        _check_paper(paper)
        cached = await db.get_cache(sess, "study_section", cache_key)

    if cached is not None:
        return {
            "paper_id":      paper_id,
            "section_title": request.section_title,
            "content":       cached.content,
            "cached":        True,
        }

    # Generate (outside the session — slow LLM call)
    try:
        content = await generate_section(
            paper_id=paper_id,
            section_title=request.section_title,
            section_description=request.section_description,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Study section failed for %s [%s]: %s",
                     paper_id, request.section_title, exc)
        raise HTTPException(status_code=500, detail=f"Failed to generate section: {exc}")

    # Store section_index in the level field so we can restore in correct order.
    # Format: "{level}:{index}" e.g. "beginner:2"
    level_with_index = f"{request.level}:{request.section_index}"

    # Atomic upsert
    async with db.session() as sess:
        await db.set_cache(
            sess,
            paper_id=paper_id,
            cache_type="study_section",
            cache_key=cache_key,
            content=content,
            level=level_with_index,
        )

    return {
        "paper_id":      paper_id,
        "section_title": request.section_title,
        "content":       content,
        "cached":        False,
    }


@router.post("/study/{paper_id}/flashcards")
async def study_flashcards(paper_id: str):
    """
    Return flashcards for a paper.
    Serves from cache instantly if available; generates and caches on first call.
    """
    async with db.session() as sess:
        paper = await db.get_paper(sess, paper_id)
        _check_paper(paper)
        cached = await db.get_cache(sess, "study_flashcards", paper_id)

    if cached is not None:
        try:
            cards = json.loads(cached.content)
            return {"paper_id": paper_id, "cards": cards, "cached": True}
        except (json.JSONDecodeError, ValueError):
            pass

    try:
        result = await generate_flashcards(paper_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Flashcard generation failed for %s: %s", paper_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to generate flashcards: {exc}")

    cards = result.get("cards", [])

    async with db.session() as sess:
        await db.set_cache(
            sess,
            paper_id=paper_id,
            cache_type="study_flashcards",
            cache_key=paper_id,
            content=json.dumps(cards),
        )

    return {"paper_id": paper_id, "cards": cards, "cached": False}


@router.get("/study/{paper_id}/cache")
async def get_study_cache_status(paper_id: str):
    """
    Return what study content is already cached for a paper.
    The frontend calls this on mount to restore session state instantly.
    """
    async with db.session() as sess:
        paper = await db.get_paper(sess, paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found.")

        outline_row     = await db.get_cache(sess, "study_outline",    paper_id)
        flashcards_row  = await db.get_cache(sess, "study_flashcards", paper_id)
        section_result  = await sess.execute(
            select(GeneratedCache).where(
                GeneratedCache.paper_id   == paper_id,
                GeneratedCache.cache_type == "study_section",
            )
        )
        section_rows = section_result.scalars().all()

    outline = None
    if outline_row is not None:
        try:
            outline = {
                "content":    json.loads(outline_row.content),
                "created_at": outline_row.created_at.isoformat(),
            }
        except (json.JSONDecodeError, ValueError):
            outline = None

    sections_raw = [
        {
            "section_title": row.cache_key.split("::", 1)[1] if "::" in row.cache_key else row.cache_key,
            "content":       row.content,
            "level":         row.level.split(":")[0] if row.level and ":" in row.level else row.level,
            # Parse the stored index from "{level}:{index}" format
            "section_index": int(row.level.split(":")[1]) if row.level and ":" in row.level else 999,
            "created_at":    row.created_at.isoformat(),
        }
        for row in section_rows
    ]
    # Sort by section_index so the frontend always gets them in outline order
    sections = sorted(sections_raw, key=lambda s: s["section_index"])

    flashcards = None
    if flashcards_row is not None:
        try:
            flashcards = {
                "cards":      json.loads(flashcards_row.content),
                "created_at": flashcards_row.created_at.isoformat(),
            }
        except (json.JSONDecodeError, ValueError):
            flashcards = None

    return {
        "paper_id":   paper_id,
        "outline":    outline,
        "sections":   sections,
        "flashcards": flashcards,
    }


@router.delete("/study/{paper_id}/cache")
async def bust_study_cache(paper_id: str):
    """
    Delete all cached study content for a paper.
    The next POST will regenerate from scratch.
    """
    from sqlalchemy import delete as sql_delete
    async with db.session() as sess:
        paper = await db.get_paper(sess, paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found.")
        result = await sess.execute(
            sql_delete(GeneratedCache).where(
                GeneratedCache.paper_id == paper_id,
                GeneratedCache.cache_type.in_(
                    ["study_outline", "study_section", "study_flashcards"]
                ),
            )
        )
        deleted = result.rowcount

    return {"success": True, "paper_id": paper_id, "deleted": deleted}