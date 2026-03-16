"""
chunker.py — Splits an ExtractedDocument into overlapping chunks.

Splits on sentence boundaries where possible.
Uses CHUNK_SIZE, CHUNK_OVERLAP, and MIN_CHUNK_CHARS from config.
"""

import logging
import re
from dataclasses import dataclass, field

from config import cfg
from processing.extractor import ExtractedDocument, Section

logger = logging.getLogger(__name__)

# Sentence boundary: period/!/? followed by whitespace and an uppercase letter,
# or a newline that acts as a natural paragraph break.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\n+")


# ── Output type ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    paper_id:    str
    chunk_index: int
    embedding_id: str           # {paper_id}_chunk_{index}
    content:     str
    metadata:    dict = field(default_factory=dict)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation and newline boundaries."""
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def _make_chunks_from_sentences(
    sentences: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Pack sentences into chunks of roughly chunk_size characters,
    with overlap characters carried over from the previous chunk.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If a single sentence exceeds chunk_size, hard-split it
        if sentence_len > chunk_size:
            # Flush current buffer first
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            # Hard-split the long sentence
            for start in range(0, sentence_len, chunk_size - overlap):
                piece = sentence[start : start + chunk_size]
                if piece:
                    chunks.append(piece)
            continue

        # Would adding this sentence exceed chunk_size?
        if current_len + sentence_len + 1 > chunk_size and current:
            chunks.append(" ".join(current))

            # Seed next chunk with overlap from the end of current
            overlap_text = " ".join(current)[-overlap:] if overlap > 0 else ""
            current = [overlap_text] if overlap_text else []
            current_len = len(overlap_text)

        current.append(sentence)
        current_len += sentence_len + 1   # +1 for the space

    if current:
        chunks.append(" ".join(current))

    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def chunk(document: ExtractedDocument) -> list[Chunk]:
    """
    Split an ExtractedDocument into Chunk objects.

    Returns a list of Chunk objects, discarding any shorter than MIN_CHUNK_CHARS.
    """
    chunk_size = cfg.CHUNK_SIZE
    overlap    = cfg.CHUNK_OVERLAP
    min_chars  = cfg.MIN_CHUNK_CHARS

    all_chunks: list[Chunk] = []
    chunk_index = 0

    for section in document.sections:
        if not section.content.strip():
            continue

        sentences   = _split_sentences(section.content)
        text_chunks = _make_chunks_from_sentences(sentences, chunk_size, overlap)

        for text in text_chunks:
            if len(text) < min_chars:
                continue

            embedding_id = f"{document.paper_id}_chunk_{chunk_index}"

            all_chunks.append(Chunk(
                paper_id=document.paper_id,
                chunk_index=chunk_index,
                embedding_id=embedding_id,
                content=text,
                metadata={
                    "paper_id":      document.paper_id,
                    "source":        document.source,
                    "section_type":  section.section_type,
                    "section_order": section.section_order,
                    "section_title": section.title,
                    "chunk_index":   chunk_index,
                },
            ))
            chunk_index += 1

    logger.info(
        "Chunking complete: paper_id=%s chunks=%d",
        document.paper_id, len(all_chunks),
    )
    return all_chunks
