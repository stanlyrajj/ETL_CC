"""
linkedin_post.py — Generate a LinkedIn post from a processed paper.
"""

import json
import logging

from database import db
from llm.factory import get_provider
from processing.embedder import embed
from processing.vector_store import VectorStore

logger = logging.getLogger(__name__)

_MAX_CONTEXT_CHARS = 4000

# Shared VectorStore — created once on first call
_vector_store: VectorStore | None = None


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(embed_fn=embed)
    return _vector_store


def _validate(raw: dict) -> dict:
    """
    Validate and normalise the LLM output into the expected structure.

    Expected: {content: str, hashtags: list[str]}
    Raises ValueError with a clear message on any structural problem.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object from LLM, got {type(raw).__name__}.")

    # Accept either "post" or "content" as the main text key
    post_text = raw.get("content") or raw.get("post") or ""
    if not isinstance(post_text, str) or not post_text.strip():
        raise ValueError("Output missing 'content' (or 'post') field or it is empty.")

    hashtags = raw.get("hashtags") or []
    if not isinstance(hashtags, list):
        hashtags = []

    return {
        "content":  post_text.strip(),
        "hashtags": [str(h) for h in hashtags],
    }


async def generate(paper_id: str, style: str, tone: str) -> dict:
    """
    Generate a LinkedIn post for the given paper.

    1. Retrieve paper context from the vector store
    2. Call the LLM to generate the post
    3. Validate output structure
    4. Save to SocialContent table
    5. Return the validated dict
    """
    # ── 1. Retrieve paper context ──────────────────────────────────────────────
    vs      = _get_vector_store()
    context = await vs.get_paper_context(paper_id, max_chars=_MAX_CONTEXT_CHARS)

    if not context.strip():
        raise ValueError(
            f"No content found for paper_id={paper_id!r}. "
            "Ensure the paper has been processed and embedded first."
        )

    # ── 2. Get paper title from database ──────────────────────────────────────
    async with db.session() as sess:
        paper = await db.get_paper(sess, paper_id)
    title = paper.title if paper else paper_id

    # ── 3. Call LLM ───────────────────────────────────────────────────────────
    provider = get_provider()
    raw      = await provider.generate_linkedin_post(
        context=context,
        title=title,
        style=style,
        tone=tone,
    )

    # ── 4. Validate output structure ──────────────────────────────────────────
    result = _validate(raw)

    # ── 5. Save to database ───────────────────────────────────────────────────
    async with db.session() as sess:
        await db.save_social(
            sess,
            paper_id=paper_id,
            platform="linkedin",
            content_type="post",
            content=result["content"],
            hashtags=result["hashtags"],
        )

    logger.info(
        "LinkedIn post generated: paper_id=%s chars=%d hashtags=%d",
        paper_id, len(result["content"]), len(result["hashtags"]),
    )
    return result