"""
vector_store.py — Persistent ChromaDB vector storage for document chunks.

Accepts the embedder as a constructor parameter — never creates a second one.
"""

import logging
from typing import Any, Callable, Coroutine

import chromadb
from chromadb.config import Settings

from config import cfg
from processing.chunker import Chunk

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "researchrag_chunks"


class VectorStore:
    def __init__(self, embed_fn: Callable[[list[Chunk]], Coroutine[Any, Any, list[list[float]]]]):
        """
        Parameters
        ----------
        embed_fn:
            The embed() coroutine from embedder.py.
            Passed in so we never instantiate a second model.
        """
        self._embed   = embed_fn
        self._client  = chromadb.PersistentClient(
            path=cfg.CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("VectorStore ready: collection=%s path=%s", _COLLECTION_NAME, cfg.CHROMA_DIR)

    async def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Upsert chunks and their embeddings into ChromaDB."""
        if not chunks:
            return

        ids        = [c.embedding_id for c in chunks]
        documents  = [c.content      for c in chunks]
        metadatas  = [c.metadata     for c in chunks]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Upserted %d chunks for paper_id=%s", len(chunks), chunks[0].paper_id)

    async def query(
        self,
        query_text: str,
        n_results:  int = 5,
        paper_id:   str | None = None,
    ) -> list[dict]:
        """
        Search for chunks similar to query_text.

        If paper_id is provided, restricts results to that paper.
        Returns a list of dicts with keys: content, metadata, distance.
        """
        # Embed the query using the shared embed function
        # We create a temporary Chunk-like object just to reuse embed()
        from processing.chunker import Chunk as _Chunk
        query_chunk = _Chunk(
            paper_id="__query__",
            chunk_index=0,
            embedding_id="__query___chunk_0",
            content=query_text,
        )
        vectors = await self._embed([query_chunk])
        query_vector = vectors[0]

        where = {"paper_id": paper_id} if paper_id else None

        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Flatten ChromaDB's nested list response
        docs      = results.get("documents",  [[]])[0]
        metas     = results.get("metadatas",  [[]])[0]
        distances = results.get("distances",  [[]])[0]

        return [
            {"content": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(docs, metas, distances)
        ]

    def delete_paper(self, paper_id: str) -> None:
        """Remove all chunks belonging to paper_id from ChromaDB."""
        self._collection.delete(where={"paper_id": paper_id})
        logger.info("Deleted all chunks for paper_id=%s", paper_id)

    async def get_paper_context(self, paper_id: str, max_chars: int = 4000) -> str:
        """
        Return concatenated chunk text for a paper, up to max_chars.

        Used as fallback context when a specific query isn't available.
        """
        results = self._collection.get(
            where={"paper_id": paper_id},
            include=["documents", "metadatas"],
        )

        docs  = results.get("documents", [])
        metas = results.get("metadatas", [])

        # Sort by section_order then chunk_index so text reads in document order
        paired = sorted(
            zip(docs, metas),
            key=lambda x: (x[1].get("section_order", 0), x[1].get("chunk_index", 0)),
        )

        combined = ""
        for doc, _ in paired:
            if len(combined) + len(doc) > max_chars:
                remaining = max_chars - len(combined)
                combined += doc[:remaining]
                break
            combined += doc + "\n\n"

        logger.debug(
            "get_paper_context: paper_id=%s returned %d chars", paper_id, len(combined)
        )
        return combined.strip()
