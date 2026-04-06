"""
study.py — Study assistant API endpoints.

POST /api/study/{paper_id}/outline    — generate learning outline for approval
POST /api/study/{paper_id}/section    — generate one section of the lesson
POST /api/study/{paper_id}/flashcards — generate flashcards after lesson completes
"""

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
    Analyze the paper and return a proposed learning sequence.
    The frontend shows this to the user for approval before teaching begins.
    """
    async with db.session() as sess:
        paper = await db.get_paper(sess, paper_id)
    _check_paper(paper)

    try:
        result = await generate_outline(paper_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Study outline failed for %s: %s", paper_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to generate outline: {exc}")

    return {"paper_id": paper_id, "outline": result}


@router.post("/study/{paper_id}/section")
async def study_section(paper_id: str, request: SectionRequest):
    """
    Generate the teaching content for one section.
    Called sequentially by the frontend as the user progresses through the lesson.
    Returns markdown content.
    """
    async with db.session() as sess:
        paper = await db.get_paper(sess, paper_id)
    _check_paper(paper)

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

    return {
        "paper_id":      paper_id,
        "section_title": request.section_title,
        "content":       content,
    }


@router.post("/study/{paper_id}/flashcards")
async def study_flashcards(paper_id: str):
    """
    Generate flashcards after the lesson completes.
    Returns a list of {front, back} card objects.
    """
    async with db.session() as sess:
        paper = await db.get_paper(sess, paper_id)
    _check_paper(paper)

    try:
        result = await generate_flashcards(paper_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Flashcard generation failed for %s: %s", paper_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to generate flashcards: {exc}")

    return {"paper_id": paper_id, "cards": result.get("cards", [])}
