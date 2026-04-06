"""
chat.py — Chat session and RAG message endpoints.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chat import session as session_ops
from chat.rag import RAGException, respond
from database import db

logger = logging.getLogger(__name__)

router = APIRouter()

_VALID_MODES  = ("standard", "study", "technical")
_VALID_LEVELS = ("beginner", "intermediate", "advanced")


# ── Pydantic models ───────────────────────────────────────────────────────────

class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    level:   str = Field("beginner")


class LevelUpdate(BaseModel):
    level: str = Field(..., pattern="^(beginner|intermediate|advanced)$")


class CreateSessionRequest(BaseModel):
    paper_id: str = Field(..., min_length=1)
    topic:    str = Field("")
    level:    str = Field("beginner")
    mode:     str = Field("standard")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/chat/sessions")
async def list_sessions():
    """Return all chat sessions ordered by last_active_at descending."""
    sessions = await session_ops.list_sessions()
    return {"sessions": sessions}


@router.post("/chat/sessions")
async def create_session(request: CreateSessionRequest):
    """
    Create a new chat session for a paper with the given mode.

    Called by the frontend when the user switches to a different chat mode
    and no existing session exists for that paper+mode combination.
    """
    mode  = request.mode  if request.mode  in _VALID_MODES  else "standard"
    level = request.level if request.level in _VALID_LEVELS else "beginner"

    async with db.session() as sess:
        paper = await db.get_paper(sess, request.paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"Paper not found: {request.paper_id!r}")

    topic = request.topic or paper.title or request.paper_id
    session_id = await session_ops.create_session(
        paper_id=request.paper_id,
        topic=topic,
        level=level,
        mode=mode,
    )
    session = await session_ops.get_session(session_id)
    return {"session": session}


@router.get("/chat/sessions/{session_id}")
async def get_session(session_id: str):
    """Return session details with full message history."""
    session = await session_ops.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id!r}")
    return {"session": session}


@router.post("/chat/sessions/{session_id}/message")
async def send_message(session_id: str, request: MessageRequest):
    """Send a message and receive a RAG-grounded assistant response."""
    session = await session_ops.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id!r}")

    level = request.level if request.level in _VALID_LEVELS else "beginner"

    try:
        response = await respond(session_id, request.message, level)
    except RAGException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not retrieve relevant content: {exc}",
        )
    except Exception as exc:
        logger.error("Unexpected error in send_message session=%s: %s", session_id, exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while generating the response.",
        )

    return {
        "session_id": session_id,
        "response":   response,
        "level":      level,
    }


@router.patch("/chat/sessions/{session_id}/level")
async def update_level(session_id: str, request: LevelUpdate):
    """Update the teaching level for a session."""
    session = await session_ops.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id!r}")

    await session_ops.update_level(session_id, request.level)
    return {
        "success":    True,
        "session_id": session_id,
        "level":      request.level,
    }


@router.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a single chat session and all its messages."""
    session = await session_ops.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id!r}")

    async with db.session() as sess:
        from sqlalchemy import delete as sql_delete
        from database import ChatSession
        result = await sess.execute(
            sql_delete(ChatSession).where(ChatSession.session_id == session_id)
        )
        deleted = result.rowcount > 0

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id!r}")

    logger.info("Deleted session %s", session_id)
    return {"success": True, "session_id": session_id, "message": "Session deleted successfully."}
