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

# Shared VectorStore — created once on first call
_vector_store: VectorStore | None = None


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(embed_fn=embed)
    return _vector_store


class CarouselValidationError(ValueError):
    pass


def _validate(raw: dict) -> dict:
    """
    Validate and normalise the LLM output into the expected structure.

    Expected:
      {
        slides: [
          {type, title, body, slide_note},   # slide_note replaces old visual_hint
          ...
        ],
        hashtags: list[str]
      }

    Validation approach:
    - Hard errors: missing slides list, non-dict slide, missing title/body
    - Soft recovery: unknown type -> "finding", wrong first/last types -> injected
    - Field aliases: heading/title, content/body, slide_note/visual_hint/speaker_note
    """
    if not isinstance(raw, dict):
        raise CarouselValidationError(
            f"Expected a JSON object from LLM, got {type(raw).__name__}."
        )

    slides_raw = raw.get("slides")
    if not isinstance(slides_raw, list) or len(slides_raw) == 0:
        raise CarouselValidationError("Output missing 'slides' list or list is empty.")

    # Type alias map — handles common LLM deviations
    _aliases: dict[str, str] = {
        "introduction":   "cover",
        "intro":          "cover",
        "title":          "cover",
        "headline":       "cover",
        "conclusion":     "cta",
        "call_to_action": "cta",
        "call to action": "cta",
        "outro":          "cta",
        "result":         "finding",
        "results":        "finding",
        "insight":        "finding",
        "key finding":    "finding",
        "methodology":    "method",
        "approach":       "method",
        "statistic":      "stat",
        "statistics":     "stat",
        "data":           "stat",
        "number":         "stat",
        "testimonial":    "quote",
    }

    slides: list[dict] = []
    for i, slide in enumerate(slides_raw):
        if not isinstance(slide, dict):
            raise CarouselValidationError(
                f"Slide {i + 1} is not a JSON object (got {type(slide).__name__})."
            )

        # Normalise title — LLM sometimes uses "heading"
        title = (
            slide.get("title") or slide.get("heading") or ""
        ).strip()

        # Normalise body — LLM sometimes uses "content" or "text"
        body = (
            slide.get("body") or slide.get("content") or slide.get("text") or ""
        ).strip()

        # Normalise slide_note — accepts old visual_hint and speaker_note too
        slide_note = (
            slide.get("slide_note")
            or slide.get("visual_hint")
            or slide.get("visual")
            or slide.get("speaker_note")
            or ""
        ).strip()

        # Reject slide_note values that are clearly color scheme bleed-through
        _color_values = {"light", "dark", "bold", "auto", "default"}
        if slide_note.lower() in _color_values:
            logger.warning(
                "Slide %d: slide_note looks like a color scheme value (%r) — clearing.",
                i + 1, slide_note,
            )
            slide_note = ""

        slide_type = str(slide.get("type") or "").lower().strip()
        slide_type = _aliases.get(slide_type, slide_type)

        if slide_type not in _VALID_SLIDE_TYPES:
            logger.warning(
                "Slide %d has unknown type %r — defaulting to 'finding'", i + 1, slide_type
            )
            slide_type = "finding"

        if not title:
            raise CarouselValidationError(f"Slide {i + 1} is missing a title.")
        if not body:
            raise CarouselValidationError(f"Slide {i + 1} is missing body text.")

        slides.append({
            "type":       slide_type,
            "title":      title,
            "body":       body,
            "slide_note": slide_note,
        })

    # ── Structural enforcement with soft recovery ─────────────────────────────

    # First slide must be cover — if not, prepend a minimal cover derived from slide 1
    if slides[0]["type"] != "cover":
        logger.warning(
            "First slide is type %r, not 'cover' — prepending a cover slide.", slides[0]["type"]
        )
        slides.insert(0, {
            "type":       "cover",
            "title":      slides[0]["title"],
            "body":       "Swipe to explore the key findings from this paper.",
            "slide_note": "",
        })

    # Last slide must be cta — if not, append a minimal cta
    if slides[-1]["type"] != "cta":
        logger.warning(
            "Last slide is type %r, not 'cta' — appending a cta slide.", slides[-1]["type"]
        )
        slides.append({
            "type":       "cta",
            "title":      "Read the full paper",
            "body":       "Find the complete methodology, results, and references in the original publication.",
            "slide_note": "",
        })

    hashtags = raw.get("hashtags") or []
    if not isinstance(hashtags, list):
        hashtags = []

    return {
        "slides":   slides,
        "hashtags": [str(h).lstrip("#") for h in hashtags],
    }


async def generate(
    paper_id:     str,
    description:  str,
    color_scheme: str,
) -> dict:
    """
    Generate LinkedIn carousel slide content for the given paper.

    1. Retrieve paper context from the vector store
    2. Call the LLM with the creator's description and color_scheme
    3. Validate output structure (cover first, cta last, all required fields)
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
    raw      = await provider.generate_carousel_content(
        context=context,
        title=title,
        description=description,
        color_scheme=color_scheme,
    )

    # ── 4. Validate output structure ──────────────────────────────────────────
    result = _validate(raw)

    # ── 5. Save to database ───────────────────────────────────────────────────
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