"""
session.py — Database operations for chat sessions and message history.
Each function does exactly one database operation.
All functions open their own session via the db singleton.
"""
import logging
import uuid
from database import db

logger = logging.getLogger(__name__)

_VALID_MODES = ("standard", "study", "technical")


async def create_session(
    paper_id: str,
    topic:    str,
    level:    str,
    mode:     str = "standard",
) -> str:
    """
    Create a new ChatSession in the database.
    Returns the new session_id.
    """
    if mode not in _VALID_MODES:
        mode = "standard"
    session_id = str(uuid.uuid4())
    async with db.session() as sess:
        await db.create_session(sess, session_id, paper_id, topic, level, mode)
    logger.info("Created session %s for paper %s (mode=%s)", session_id, paper_id, mode)
    return session_id


async def get_session(session_id: str) -> dict | None:
    """
    Return session details with full message history.
    Returns None if the session does not exist.
    """
    async with db.session() as sess:
        chat_session = await db.get_session(sess, session_id)
        if chat_session is None:
            return None
        messages = await db.get_messages(sess, session_id)
    return {
        "session_id":     chat_session.session_id,
        "paper_id":       chat_session.paper_id,
        "topic":          chat_session.topic,
        "level":          chat_session.level,
        "mode":           chat_session.mode or "standard",
        "title":          chat_session.title,
        "last_active_at": chat_session.last_active_at.isoformat() if chat_session.last_active_at else None,
        "created_at":     chat_session.created_at.isoformat()     if chat_session.created_at     else None,
        "messages": [
            {
                "id":         msg.id,
                "role":       msg.role,
                "content":    msg.content,
                "level":      msg.level,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
            }
            for msg in messages
        ],
    }


async def list_sessions() -> list[dict]:
    """
    Return all sessions ordered by last_active_at descending.
    """
    async with db.session() as sess:
        sessions = await db.list_sessions(sess)
    return [
        {
            "session_id":     s.session_id,
            "paper_id":       s.paper_id,
            "topic":          s.topic,
            "level":          s.level,
            "mode":           s.mode or "standard",
            "title":          s.title,
            "last_active_at": s.last_active_at.isoformat() if s.last_active_at else None,
            "created_at":     s.created_at.isoformat()     if s.created_at     else None,
        }
        for s in sessions
    ]


async def update_level(session_id: str, level: str) -> None:
    """Update the teaching level for a session."""
    async with db.session() as sess:
        await db.update_session_level(sess, session_id, level)
    logger.info("Updated session %s level to %s", session_id, level)


async def add_message(session_id: str, role: str, content: str, level: str) -> None:
    """Save a message to the session and update last_active_at."""
    async with db.session() as sess:
        await db.add_message(sess, session_id, role, content, level)
