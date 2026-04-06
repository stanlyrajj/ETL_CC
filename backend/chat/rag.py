"""
rag.py — Retrieval-augmented chat response.

respond() is the single entry point. It retrieves relevant chunks,
builds context, calls the LLM with the correct mode prompt,
saves the exchange, and returns the response.
"""

import logging

from database import db
from chat.session import get_session
from llm.factory import get_provider
from processing.vector_store import VectorStore
from processing.embedder import embed

logger = logging.getLogger(__name__)

_MAX_CONTEXT_CHARS  = 4000
_MAX_HISTORY_MSGS   = 20

_vector_store: VectorStore | None = None


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(embed_fn=embed)
    return _vector_store


class RAGException(Exception):
    pass


async def respond(session_id: str, message: str, level: str) -> str:
    """
    Generate a grounded response to a user message.

    The session's mode field controls which system prompt is used:
      standard  — general research assistant
      study     — flashcards, questions, examples
      technical — system design, API structures, scalability
    """
    # ── 1. Load session ───────────────────────────────────────────────────────
    session = await get_session(session_id)
    if session is None:
        raise RAGException(f"Session not found: {session_id!r}")

    paper_id = session["paper_id"]
    mode     = session.get("mode", "standard")

    # ── 2. Retrieve relevant chunks ───────────────────────────────────────────
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

    if not results:
        raise RAGException(
            f"No relevant content found for paper_id={paper_id!r}. "
            "Ensure the paper has been processed and embedded before chatting."
        )

    # ── 3. Build context ──────────────────────────────────────────────────────
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

    if not context.strip():
        raise RAGException(
            f"Retrieved chunks contain no usable content for paper_id={paper_id!r}. "
            "The paper may need to be re-processed."
        )

    # ── 4. Truncate history ───────────────────────────────────────────────────
    all_messages = session["messages"]
    if len(all_messages) > _MAX_HISTORY_MSGS:
        all_messages = all_messages[-_MAX_HISTORY_MSGS:]

    history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in all_messages
    ]

    # ── 5. Call LLM with mode-aware system prompt ─────────────────────────────
    provider = get_provider()
    title    = session.get("title") or session.get("topic") or "Research Paper"

    response = await provider.chat_response(
        context=context,
        title=title,
        authors=[],
        history=history,
        message=message,
        level=level,
        mode=mode,
    )

    # ── 6. Save atomically ────────────────────────────────────────────────────
    async with db.session() as sess:
        await db.add_message(sess, session_id, role="user",      content=message,  level=level)
        await db.add_message(sess, session_id, role="assistant", content=response, level=level)

    logger.info(
        "RAG response: session=%s paper=%s level=%s mode=%s context_chars=%d response_chars=%d",
        session_id, paper_id, level, mode, len(context), len(response),
    )

    return response
