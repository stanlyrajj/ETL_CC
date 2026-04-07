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

_vector_store: VectorStore | None = None


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(embed_fn=embed)
    return _vector_store


def _validate(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object from LLM, got {type(raw).__name__}.")

    post_text = raw.get("content") or raw.get("post") or ""
    if not isinstance(post_text, str) or not post_text.strip():
        raise ValueError("Output missing 'content' (or 'post') field or it is empty.")

    hashtags = raw.get("hashtags") or []
    if not isinstance(hashtags, list):
        hashtags = []

    # Pass through inferred_attributes if present — stored in content metadata
    inferred = raw.get("inferred_attributes") or {}

    return {
        "content":             post_text.strip(),
        "hook":                raw.get("hook", ""),
        "hashtags":            [str(h) for h in hashtags],
        "inferred_attributes": inferred,
    }


async def generate(paper_id: str, description: str) -> dict:
    """
    Generate a LinkedIn post for the given paper.

    description: user's one-line brief describing intent, audience, tone.
                 All content attributes are inferred from it by the LLM.
    """
    vs      = _get_vector_store()
    context = await vs.get_paper_context(paper_id, max_chars=_MAX_CONTEXT_CHARS)

    if not context.strip():
        raise ValueError(
            f"No content found for paper_id={paper_id!r}. "
            "Ensure the paper has been processed and embedded first."
        )

    async with db.session() as sess:
        paper = await db.get_paper(sess, paper_id)
    title = paper.title if paper else paper_id

    provider = get_provider()
    raw      = await provider.generate_linkedin_post(
        context=context,
        title=title,
        description=description,
    )

    result = _validate(raw)

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
