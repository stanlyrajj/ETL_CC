"""
rag.py — Retrieval-augmented chat response.

respond() is the single entry point. It retrieves relevant chunks,
builds context, calls the LLM, saves the exchange, and returns the response.
"""

import logging

from database import db
from chat.session import get_session
from llm.factory import get_provider
from processing.vector_store import VectorStore
from processing.embedder import embed

logger = logging.getLogger(__name__)

_MAX_CONTEXT_CHARS  = 4000
_MAX_HISTORY_MSGS   = 20   # keep last N messages to prevent context overflow

# Shared VectorStore instance — created once on first call
_vector_store: VectorStore | None = None


def _get_vector_store() -> VectorStore:
    """Return the shared VectorStore, creating it on first call."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(embed_fn=embed)
    return _vector_store


class RAGException(Exception):
    pass


async def respond(session_id: str, message: str, level: str) -> str:
    """
    Generate a grounded response to a user message.

    Steps (in order):
    1. Retrieve top 5 relevant chunks from ChromaDB
    2. Raise RAGException if retrieval fails or returns nothing
    3. Build a context string from retrieved chunks, max 4000 chars
    4. Raise RAGException if context is empty after assembly
    5. Get message history for the session (truncated to last 20)
    6. Call the LLM with context, history, message, and level
    7. Save both user message and assistant response atomically
    8. Return the assistant response string
    """
    # ── 1. Retrieve top 5 relevant chunks ─────────────────────────────────────
    session = await get_session(session_id)
    if session is None:
        raise RAGException(f"Session not found: {session_id!r}")

    paper_id = session["paper_id"]

    try:
        vs = _get_vector_store()
        results = await vs.query(
            query_text=message,
            n_results=5,
            paper_id=paper_id,
        )
    except Exception as exc:
        raise RAGException(
            f"Vector store retrieval failed for paper_id={paper_id!r}: {exc}"
        ) from exc

    # ── 2. Raise if retrieval returned nothing ─────────────────────────────────
    if not results:
        raise RAGException(
            f"No relevant content found for paper_id={paper_id!r}. "
            "Ensure the paper has been processed and embedded before chatting."
        )

    # ── 3. Build context string, truncated to _MAX_CONTEXT_CHARS ──────────────
    context_parts = []
    total = 0
    for result in results:
        chunk_text = result["content"]
        if total + len(chunk_text) > _MAX_CONTEXT_CHARS:
            remaining = _MAX_CONTEXT_CHARS - total
            if remaining > 0:
                context_parts.append(chunk_text[:remaining])
            break
        context_parts.append(chunk_text)
        total += len(chunk_text)

    context = "\n\n".join(context_parts)

    # ── 4. Raise if context is empty after assembly ────────────────────────────
    # Guard against results whose content fields are all empty strings.
    if not context.strip():
        raise RAGException(
            f"Retrieved chunks contain no usable content for paper_id={paper_id!r}. "
            "The paper may need to be re-processed."
        )

    # ── 5. Get message history, truncated to last _MAX_HISTORY_MSGS ───────────
    # Truncate before building the list to prevent LLM context overflow.
    all_messages = session["messages"]
    if len(all_messages) > _MAX_HISTORY_MSGS:
        all_messages = all_messages[-_MAX_HISTORY_MSGS:]

    history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in all_messages
    ]

    # ── 6. Call the LLM ───────────────────────────────────────────────────────
    provider = get_provider()

    # FIX: Resolve title correctly — prefer session["title"] (set from paper.title
    # after extraction), fall back to session["topic"], then a safe default.
    # Previously only session["topic"] was used, so the LLM prompt always showed
    # the search topic string instead of the paper's actual title.
    title = session.get("title") or session.get("topic") or "Research Paper"
    authors: list[str] = []

    response = await provider.chat_response(
        context=context,
        title=title,
        authors=authors,
        history=history,
        message=message,
        level=level,
    )

    # ── 7. Save user message and assistant response atomically ────────────────
    # Both writes share a single db.session() so a crash between them cannot
    # leave an orphaned user message with no matching assistant reply.
    async with db.session() as sess:
        await db.add_message(sess, session_id, role="user",      content=message,  level=level)
        await db.add_message(sess, session_id, role="assistant", content=response, level=level)

    logger.info(
        "RAG response: session=%s paper=%s level=%s context_chars=%d response_chars=%d",
        session_id, paper_id, level, len(context), len(response),
    )

    # ── 8. Return the response ────────────────────────────────────────────────
    return response