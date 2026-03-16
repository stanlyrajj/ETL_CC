"""
twitter.py — Generate a Twitter/X thread from a processed paper.
"""

import logging

from database import db
from llm.factory import get_provider
from processing.embedder import embed
from processing.vector_store import VectorStore

logger = logging.getLogger(__name__)

_MAX_CONTEXT_CHARS = 4000
_MAX_TWEET_CHARS   = 280

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

    Expected: {tweets: list[str|dict], hashtags: list[str]}
    Raises ValueError with a clear message on any structural problem.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object from LLM, got {type(raw).__name__}.")

    tweets_raw = raw.get("tweets")
    if not isinstance(tweets_raw, list) or len(tweets_raw) == 0:
        raise ValueError("Output missing 'tweets' list or list is empty.")

    # Normalise: LLM may return list of strings or list of {index, content} dicts
    tweets: list[str] = []
    for item in tweets_raw:
        if isinstance(item, str):
            tweets.append(item.strip())
        elif isinstance(item, dict):
            text = item.get("content") or item.get("text") or ""
            tweets.append(str(text).strip())
        else:
            tweets.append(str(item).strip())

    # Filter empty entries
    tweets = [t for t in tweets if t]
    if not tweets:
        raise ValueError("All tweets were empty after normalisation.")

    # Truncate any tweet that exceeds 280 characters
    truncated = 0
    capped: list[str] = []
    for t in tweets:
        if len(t) > _MAX_TWEET_CHARS:
            capped.append(t[:_MAX_TWEET_CHARS])
            truncated += 1
        else:
            capped.append(t)
    if truncated:
        logger.warning("Truncated %d tweet(s) to %d chars.", truncated, _MAX_TWEET_CHARS)

    hashtags = raw.get("hashtags") or []
    if not isinstance(hashtags, list):
        hashtags = []

    return {"tweets": capped, "hashtags": [str(h) for h in hashtags]}


async def generate(paper_id: str, style: str, tone: str) -> dict:
    """
    Generate a Twitter/X thread for the given paper.

    1. Retrieve paper context from the vector store
    2. Call the LLM to generate the thread
    3. Validate output structure
    4. Enforce 280-char limit per tweet
    5. Save to SocialContent table
    6. Return the validated dict
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
    raw      = await provider.generate_twitter_thread(
        context=context,
        title=title,
        style=style,
        tone=tone,
    )

    # ── 4. Validate output structure ──────────────────────────────────────────
    result = _validate(raw)

    # ── 5. Save to database ───────────────────────────────────────────────────
    import json
    async with db.session() as sess:
        await db.save_social(
            sess,
            paper_id=paper_id,
            platform="twitter",
            content_type="thread",
            content=json.dumps(result["tweets"]),
            hashtags=result["hashtags"],
        )

    logger.info(
        "Twitter thread generated: paper_id=%s tweets=%d hashtags=%d",
        paper_id, len(result["tweets"]), len(result["hashtags"]),
    )
    return result