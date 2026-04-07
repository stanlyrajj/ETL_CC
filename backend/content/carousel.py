"""
carousel.py — Generate LinkedIn carousel slide content from a processed paper.
"""

import json
import logging

from database import db
from llm.factory import get_provider
from processing.embedder import embed
from processing.vector_store import VectorStore

logger = logging.getLogger(__name__)

_MAX_CONTEXT_CHARS = 4000
_VALID_SLIDE_TYPES = {"cover", "finding", "method", "stat", "quote", "cta"}

_vector_store: VectorStore | None = None


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(embed_fn=embed)
    return _vector_store


class CarouselValidationError(ValueError):
    pass


def _validate(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise CarouselValidationError(
            f"Expected a JSON object from LLM, got {type(raw).__name__}."
        )

    slides_raw = raw.get("slides")
    if not isinstance(slides_raw, list) or len(slides_raw) == 0:
        raise CarouselValidationError("Output missing 'slides' list or list is empty.")

    slides: list[dict] = []
    for i, slide in enumerate(slides_raw):
        if not isinstance(slide, dict):
            raise CarouselValidationError(
                f"Slide {i + 1} is not a JSON object (got {type(slide).__name__})."
            )

        title       = slide.get("title")   or slide.get("heading") or ""
        body        = slide.get("body")    or slide.get("content") or ""
        visual_hint = (
            slide.get("visual_hint") or slide.get("visual")
            or slide.get("speaker_note") or ""
        )

        slide_type = str(slide.get("type") or "").lower().strip()

        _aliases = {
            "introduction": "cover", "intro": "cover", "title": "cover",
            "conclusion": "cta", "call_to_action": "cta", "call to action": "cta",
            "result": "finding", "results": "finding",
            "methodology": "method",
            "statistic": "stat", "statistics": "stat", "data": "stat",
        }
        slide_type = _aliases.get(slide_type, slide_type)

        if slide_type not in _VALID_SLIDE_TYPES:
            logger.warning(
                "Slide %d has unknown type %r — defaulting to 'finding'", i + 1, slide_type
            )
            slide_type = "finding"

        if not title.strip():
            raise CarouselValidationError(f"Slide {i + 1} is missing a title.")
        if not body.strip():
            raise CarouselValidationError(f"Slide {i + 1} is missing body text.")

        slides.append({
            "type":        slide_type,
            "title":       title.strip(),
            "body":        body.strip(),
            "visual_hint": visual_hint.strip(),
        })

    if slides[0]["type"] != "cover":
        raise CarouselValidationError(
            f"First slide must be type 'cover', got {slides[0]['type']!r}. "
            "Check the LLM prompt or regenerate."
        )

    if slides[-1]["type"] != "cta":
        raise CarouselValidationError(
            f"Last slide must be type 'cta', got {slides[-1]['type']!r}. "
            "Check the LLM prompt or regenerate."
        )

    hashtags = raw.get("hashtags") or []
    if not isinstance(hashtags, list):
        hashtags = []

    return {
        "slides":   slides,
        "hashtags": [str(h) for h in hashtags],
    }


async def generate(
    paper_id:     str,
    description:  str,
    color_scheme: str,
) -> dict:
    """
    Generate LinkedIn carousel slide content for the given paper.

    description: user's one-line brief — tone, audience, style inferred from it.
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
    raw      = await provider.generate_carousel_content(
        context=context,
        title=title,
        description=description,
        color_scheme=color_scheme,
    )

    result = _validate(raw)

    async with db.session() as sess:
        await db.save_social(
            sess,
            paper_id=paper_id,
            platform="carousel",
            content_type="slides",
            content=json.dumps(result["slides"]),
            hashtags=result["hashtags"],
        )

    logger.info(
        "Carousel generated: paper_id=%s slides=%d hashtags=%d",
        paper_id, len(result["slides"]), len(result["hashtags"]),
    )
    return result
