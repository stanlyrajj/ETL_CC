"""
study.py — Study assistant API endpoints.

POST /api/study/{paper_id}/outline    — generate (or return cached) learning outline
POST /api/study/{paper_id}/section    — generate (or return cached) one section
POST /api/study/{paper_id}/flashcards — generate (or return cached) flashcards
DELETE /api/study/{paper_id}/cache    — bust all study cache for a paper (regenerate)

Cache strategy
--------------
All generated content is stored in the generated_cache table keyed by:
  outline    → cache_type="study_outline",    cache_key=paper_id
  section    → cache_type="study_section",    cache_key="{paper_id}::{section_title}"
  flashcards → cache_type="study_flashcards", cache_key=paper_id

A POST always returns the cached version instantly if it exists.
DELETE /cache busts everything for that paper so the next POST regenerates.

The 'level' column records what level was active when the content was generated —
it is informational only and does not affect cache lookup (one entry per section).
"""

import json
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from database import db
from content.study import generate_outline, generate_section, generate_flashcards

logger = logging.getLogger(__name__)

router = APIRouter()


class SectionRequest(BaseModel):
    section_title:       str = Field(..., min_length=1)
    section_description: str = Field("")
    level:               str = Field("beginner")


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
    Serves from cache if available; generates and caches on first call.
    """
    async with db.session() as sess:
        paper = await db.get_paper(sess, paper_id)
    _check_paper(paper)

    # Check cache
    async with db.session() as sess:
        cached = await db.get_cache(sess, "study_outline", paper_id)
    if cached is not None:
        try:
            outline = json.loads(cached.content)
            return {"paper_id": paper_id, "outline": outline, "cached": True}
        except (json.JSONDecodeError, ValueError):
            pass  # corrupt cache — fall through to regenerate

    # Generate and cache
    try:
        result = await generate_outline(paper_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Study outline failed for %s: %s", paper_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to generate outline: {exc}")

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
    Serves from cache if available; generates and caches on first call.
    """
    async with db.session() as sess:
        paper = await db.get_paper(sess, paper_id)
    _check_paper(paper)

    cache_key = f"{paper_id}::{request.section_title}"

    # Check cache
    async with db.session() as sess:
        cached = await db.get_cache(sess, "study_section", cache_key)
    if cached is not None:
        return {
            "paper_id":      paper_id,
            "section_title": request.section_title,
            "content":       cached.content,
            "cached":        True,
        }

    # Generate and cache
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

    async with db.session() as sess:
        await db.set_cache(
            sess,
            paper_id=paper_id,
            cache_type="study_section",
            cache_key=cache_key,
            content=content,
            level=request.level,
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
    Serves from cache if available; generates and caches on first call.
    """
    async with db.session() as sess:
        paper = await db.get_paper(sess, paper_id)
    _check_paper(paper)

    # Check cache
    async with db.session() as sess:
        cached = await db.get_cache(sess, "study_flashcards", paper_id)
    if cached is not None:
        try:
            cards = json.loads(cached.content)
            return {"paper_id": paper_id, "cards": cards, "cached": True}
        except (json.JSONDecodeError, ValueError):
            pass

    # Generate and cache
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


@router.delete("/study/{paper_id}/cache")
async def bust_study_cache(paper_id: str):
    """
    Delete all cached study content for a paper so the next request regenerates.
    Called by the frontend 'Regenerate' button.
    """
    async with db.session() as sess:
        paper = await db.get_paper(sess, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found.")

    async with db.session() as sess:
        deleted = await db.delete_cache_for_paper(sess, paper_id, cache_type=None)
        # Delete only study-related cache types
        from sqlalchemy import delete as sql_delete
        from database import GeneratedCache
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


@router.get("/study/{paper_id}/cache")
async def get_study_cache_status(paper_id: str):
    """
    Return what study content is already cached for a paper.
    The frontend calls this on mount to decide whether to show cached content
    immediately or prompt the user to generate.

    Response:
    {
      "outline":    { content, created_at } | null,
      "sections":   [ { section_title, content, created_at }, ... ],
      "flashcards": [ {front, back}, ... ] | null
    }
    """
    async with db.session() as sess:
        paper = await db.get_paper(sess, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found.")

    async with db.session() as sess:
        # Outline
        outline_row = await db.get_cache(sess, "study_outline", paper_id)
        outline = None
        if outline_row is not None:
            try:
                outline = {
                    "content":    json.loads(outline_row.content),
                    "created_at": outline_row.created_at.isoformat(),
                }
            except (json.JSONDecodeError, ValueError):
                outline = None

        # Sections
        from sqlalchemy import select
        from database import GeneratedCache
        result = await sess.execute(
            select(GeneratedCache).where(
                GeneratedCache.paper_id   == paper_id,
                GeneratedCache.cache_type == "study_section",
            )
        )
        section_rows = result.scalars().all()
        sections = [
            {
                "section_title": row.cache_key.split("::", 1)[1] if "::" in row.cache_key else row.cache_key,
                "content":       row.content,
                "level":         row.level,
                "created_at":    row.created_at.isoformat(),
            }
            for row in section_rows
        ]

        # Flashcards
        flashcards_row = await db.get_cache(sess, "study_flashcards", paper_id)
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
        "paper_id":  paper_id,
        "outline":   outline,
        "sections":  sections,
        "flashcards": flashcards,
    }