"""
technical.py — Technical analysis content generation.

Generates five sections sequentially:
  overview → concepts → architecture → implementation → scalability

Each section is generated independently with prior sections passed as context
so the LLM maintains continuity without repeating itself.
"""

import logging

from database import db
from llm.factory import get_provider
from processing.embedder import embed
from processing.vector_store import VectorStore

logger = logging.getLogger(__name__)

_MAX_CONTEXT_CHARS = 4000

# Ordered section definitions — key must match _TECHNICAL_SECTION_PROMPTS in openai.py
TECHNICAL_SECTIONS = [
    {"key": "overview",        "label": "Overview"},
    {"key": "concepts",        "label": "Core Concepts"},
    {"key": "architecture",    "label": "System Architecture"},
    {"key": "implementation",  "label": "Implementation Details"},
    {"key": "scalability",     "label": "Scalability & Trade-offs"},
]

_vector_store: VectorStore | None = None


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(embed_fn=embed)
    return _vector_store


async def generate_section(
    paper_id:      str,
    section_key:   str,
    section_label: str,
    prev_sections: str,
) -> str:
    """
    Generate one technical analysis section.

    paper_id      : the paper to analyze
    section_key   : one of overview | concepts | architecture | implementation | scalability
    section_label : human-readable label for logging
    prev_sections : markdown text of already-generated sections (passed for continuity)

    Returns a markdown string (may include mermaid fenced blocks).
    """
    vs = _get_vector_store()
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
    content  = await provider.generate_technical_section(
        context=context,
        title=title,
        section_key=section_key,
        section_label=section_label,
        prev_sections=prev_sections,
    )

    if not content or not content.strip():
        raise ValueError(f"LLM returned empty content for technical section: {section_label!r}")

    logger.info("Technical section generated: paper_id=%s section=%s chars=%d",
                paper_id, section_key, len(content))
    return content
