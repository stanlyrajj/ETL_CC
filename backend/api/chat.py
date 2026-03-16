"""
chat.py — Chat session and RAG message endpoints.

Every route returns explicit success or failure — no silent failures.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chat import session as session_ops
from chat.rag import RAGException, respond

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Pydantic models ───────────────────────────────────────────────────────────

class MessageRequest(BaseModel):
    message: str   = Field(..., min_length=1, max_length=2000)
    level:   str   = Field("beginner")   # beginner | intermediate | advanced


class LevelUpdate(BaseModel):
    level: str = Field(..., pattern="^(beginner|intermediate|advanced)$")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/chat/sessions")
async def list_sessions():
    """Return all chat sessions ordered by last_active_at descending."""
    sessions = await session_ops.list_sessions()
    return {"sessions": sessions}


@router.get("/chat/sessions/{session_id}")
async def get_session(session_id: str):
    """
    Return session details with full message history.
    Returns 404 if the session does not exist.
    """
    session = await session_ops.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id!r}",
        )
    return {"session": session}


@router.post("/chat/sessions/{session_id}/message")
async def send_message(session_id: str, request: MessageRequest):
    """
    Send a message and receive a RAG-grounded assistant response.
    Returns 404 if the session does not exist.
    Returns 503 if RAG retrieval fails (no chunks indexed yet).
    """
    # Check session exists first for a clear 404
    session = await session_ops.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id!r}",
        )

    level = request.level if request.level in ("beginner", "intermediate", "advanced") else "beginner"

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
    """
    Update the teaching level for a session.
    Returns 404 if the session does not exist.
    """
    session = await session_ops.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id!r}",
        )

    await session_ops.update_level(session_id, request.level)
    return {
        "success":    True,
        "session_id": session_id,
        "level":      request.level,
        "message":    f"Level updated to {request.level!r}.",
    }
