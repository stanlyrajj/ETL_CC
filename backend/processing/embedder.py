"""
embedder.py — Generates embeddings for chunks using sentence-transformers.

The model is loaded lazily on the first call to embed().
Model encoding is blocking, so it runs in run_in_executor.
"""

import asyncio
import logging
from typing import Any

from config import cfg
from processing.chunker import Chunk

logger = logging.getLogger(__name__)

_MAX_BATCH_SIZE = 50    # fallback if MAX_EMBEDDING_BATCH_SIZE not in cfg

# Module-level model cache — loaded once on first use
_model: Any = None


def _get_model() -> Any:
    """Load and cache the sentence-transformers model (blocking)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", cfg.EMBEDDING_MODEL)
        _model = SentenceTransformer(cfg.EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _model


def _encode_batch(texts: list[str]) -> list[list[float]]:
    """Encode a batch of texts into embeddings (blocking)."""
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False)
    return [v.tolist() for v in vectors]


async def embed(chunks: list[Chunk]) -> list[list[float]]:
    """
    Generate embeddings for a list of chunks.

    Processes chunks in batches. Returns embeddings in the same order as input.
    """
    if not chunks:
        return []

    batch_size = getattr(cfg, "MAX_EMBEDDING_BATCH_SIZE", _MAX_BATCH_SIZE)

    # FIX E3: Use get_running_loop() — get_event_loop() is deprecated in
    # Python 3.10+ and may raise RuntimeError inside coroutines in Python 3.12.
    loop = asyncio.get_running_loop()
    all_vectors: list[list[float]] = []

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.content for c in batch]

        logger.debug(
            "Embedding batch %d-%d of %d chunks",
            start, start + len(batch) - 1, len(chunks),
        )

        vectors = await loop.run_in_executor(None, _encode_batch, texts)
        all_vectors.extend(vectors)

    logger.info("Embedding complete: %d chunks embedded", len(all_vectors))
    return all_vectors