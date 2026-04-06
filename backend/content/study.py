"""
study.py — Study assistant content generation.

Three-phase flow:
  1. generate_outline()  — analyze paper, return learning sections for user approval
  2. generate_section()  — teach one section at a time (called sequentially)
  3. generate_flashcards() — produce flip-card Q&A pairs after lesson completes
"""

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


async def _get_context_and_title(paper_id: str) -> tuple[str, str]:
    """Retrieve paper context from vector store and title from database."""
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
    return context, title


async def generate_outline(paper_id: str) -> dict:
    """
    Analyze the paper and return a structured learning outline.

    Returns:
    {
      "summary": "...",
      "sections": [{"index": 0, "title": "...", "description": "..."}, ...]
    }
    """
    context, title = await _get_context_and_title(paper_id)
    provider = get_provider()
    result   = await provider.generate_study_outline(context=context, title=title)

    # Validate
    if not result or not isinstance(result.get("sections"), list) or not result["sections"]:
        raise ValueError("LLM returned an invalid outline structure. Please try again.")

    logger.info("Study outline generated: paper_id=%s sections=%d",
                paper_id, len(result["sections"]))
    return result


async def generate_section(
    paper_id:            str,
    section_title:       str,
    section_description: str,
) -> str:
    """
    Generate the teaching content for one section.
    Returns a markdown string.
    """
    context, title = await _get_context_and_title(paper_id)
    provider = get_provider()
    content  = await provider.generate_study_section(
        context=context,
        title=title,
        section_title=section_title,
        section_description=section_description,
    )
    if not content or not content.strip():
        raise ValueError(f"LLM returned empty content for section: {section_title!r}")

    logger.info("Study section generated: paper_id=%s section=%r chars=%d",
                paper_id, section_title, len(content))
    return content


async def generate_flashcards(paper_id: str) -> dict:
    """
    Generate flashcards after the lesson completes.

    Returns:
    {
      "cards": [{"front": "...", "back": "..."}, ...]
    }
    """
    context, title = await _get_context_and_title(paper_id)
    provider = get_provider()
    result   = await provider.generate_flashcards(context=context, title=title)

    if not result or not isinstance(result.get("cards"), list) or not result["cards"]:
        raise ValueError("LLM returned invalid flashcard structure. Please try again.")

    logger.info("Flashcards generated: paper_id=%s cards=%d",
                paper_id, len(result["cards"]))
    return result
