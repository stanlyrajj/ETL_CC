"""
processor.py — Paper extraction, chunking, embedding, and vector storage.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import cfg

logger = logging.getLogger(__name__)


# ── Retry decorator ───────────────────────────────────────────────────────────

def with_retry(max_attempts: int = None, backoff_base: float = None):
    """
    Sync retry decorator with exponential backoff.

    PERF-02: PERMANENT_ERRORS are re-raised immediately without sleeping.
    Only transient errors (network I/O, timeouts) are retried.
    """
    _max = max_attempts or cfg.MAX_RETRIES
    _base = backoff_base or cfg.RETRY_BACKOFF_BASE

    # PERF-02: data errors are permanent — retrying never helps
    _PERMANENT = (ValueError, json.JSONDecodeError, UnicodeDecodeError)

    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, _max + 1):
                try:
                    return func(*args, **kwargs)
                except _PERMANENT:
                    raise  # immediate re-raise — no sleep, no retry
                except Exception as exc:
                    last_exc = exc
                    if attempt < _max:
                        wait = _base ** (attempt - 1)
                        logger.warning(
                            f"[retry] {func.__name__} attempt {attempt}/{_max} failed: {exc}. "
                            f"Retrying in {wait}s…"
                        )
                        time.sleep(wait)
            raise last_exc
        return wrapper
    return decorator


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Section:
    section_type: str
    title: str
    content: str
    order: int


@dataclass
class ExtractedPaper:
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    sections: List[Section]
    source: str

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.content for s in self.sections)


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── PDF Extractor ─────────────────────────────────────────────────────────────

class PDFExtractor:
    SECTION_PATTERNS = [
        r"(?i)^(abstract)\s*$",
        r"(?i)^(\d+\.?\s*introduction)\s*$",
        r"(?i)^(\d+\.?\s*related work)\s*$",
        r"(?i)^(\d+\.?\s*method(?:s|ology)?)\s*$",
        r"(?i)^(\d+\.?\s*experiment(?:s|al setup)?)\s*$",
        r"(?i)^(\d+\.?\s*result(?:s)?)\s*$",
        r"(?i)^(\d+\.?\s*discussion)\s*$",
        r"(?i)^(\d+\.?\s*conclusion(?:s)?)\s*$",
        r"(?i)^(references)\s*$",
    ]

    @with_retry()
    def extract(self, pdf_path: str) -> ExtractedPaper:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF not installed: pip install pymupdf")

        paper_id = Path(pdf_path).stem
        doc = fitz.open(pdf_path)

        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n"
        doc.close()

        full_text = self._clean_text(full_text)

        if len(full_text) < 100:
            raise ValueError(
                f"PDF '{pdf_path}' yielded only {len(full_text)} characters. "
                "This is likely a scanned/image PDF. OCR is required but not available."
            )

        sections = self._detect_sections(full_text)
        title, authors = self._extract_meta(full_text)
        abstract = self._extract_abstract(full_text, sections)

        return ExtractedPaper(
            paper_id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract,
            sections=sections,
            source="arxiv",
        )

    def _clean_text(self, text: str) -> str:
        import ftfy
        text = ftfy.fix_text(text)
        ligatures = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
        for lig, rep in ligatures.items():
            text = text.replace(lig, rep)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def _detect_sections(self, text: str) -> List[Section]:
        lines = text.split("\n")
        sections: List[Section] = []
        current_type = "preamble"
        current_title = ""
        current_lines: List[str] = []
        order = 0

        for line in lines:
            stripped = line.strip()
            matched = False
            for pat in self.SECTION_PATTERNS:
                if re.match(pat, stripped):
                    if current_lines:
                        content = "\n".join(current_lines).strip()
                        if len(content) >= cfg.MIN_CHUNK_CHARS:
                            sections.append(Section(
                                section_type=current_type,
                                title=current_title,
                                content=content,
                                order=order,
                            ))
                            order += 1
                    current_type = re.sub(r"^\d+\.?\s*", "", stripped).lower()
                    current_title = stripped
                    current_lines = []
                    matched = True
                    break
            if not matched:
                current_lines.append(line)

        if current_lines:
            content = "\n".join(current_lines).strip()
            if len(content) >= cfg.MIN_CHUNK_CHARS:
                sections.append(Section(
                    section_type=current_type,
                    title=current_title,
                    content=content,
                    order=order,
                ))

        return sections if sections else [Section("body", "", text, 0)]

    def _extract_meta(self, text: str):
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        title = lines[0] if lines else "Unknown Title"
        authors = []
        for line in lines[1:5]:
            if re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", line) and len(line) < 200:
                authors.append(line)
        return title, authors

    def _extract_abstract(self, text: str, sections: List[Section]) -> str:
        for s in sections:
            if "abstract" in s.section_type.lower():
                return s.content[:1000]
        return text[:500]


# ── PubMed JSON Extractor ─────────────────────────────────────────────────────

class PubMedExtractor:
    @with_retry()
    def extract(self, json_path: str) -> ExtractedPaper:
        paper_id = Path(json_path).stem
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = data.get("documents", data.get("PubTator3", []))
        if not documents:
            documents = [data]

        sections: List[Section] = []
        title = "Unknown Title"
        abstract = ""
        authors = []
        order = 0

        for doc in documents[:1]:
            passages = doc.get("passages", [])
            if not passages:
                raise ValueError(f"PubMed JSON '{json_path}' has no passages. Unusual format.")

            for passage in passages:
                infons = passage.get("infons", {})
                ptype = infons.get("type", infons.get("section_type", "body")).lower()
                text = passage.get("text", "").strip()

                if not text or len(text) < cfg.MIN_CHUNK_CHARS:
                    continue

                if ptype in ("title",):
                    title = text
                elif ptype in ("abstract",):
                    abstract = text
                    sections.append(Section("abstract", "Abstract", text, order))
                    order += 1
                else:
                    sections.append(Section(ptype, ptype.title(), text, order))
                    order += 1

            authors_raw = doc.get("authors", [])
            if isinstance(authors_raw, list):
                authors = [
                    a if isinstance(a, str) else f"{a.get('lastname', '')} {a.get('firstname', '')}".strip()
                    for a in authors_raw[:5]
                ]

        if not sections:
            raise ValueError(f"PubMed JSON '{json_path}' produced no usable sections.")

        return ExtractedPaper(
            paper_id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract or (sections[0].content[:500] if sections else ""),
            sections=sections,
            source="pubmed",
        )


# ── Text Chunker ──────────────────────────────────────────────────────────────

class TextChunker:
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or cfg.CHUNK_SIZE
        self.overlap = overlap or cfg.CHUNK_OVERLAP

    def chunk_sections(self, paper_id: str, sections: List[Section]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for section in sections:
            section_chunks = self._chunk_text(section.content)
            for i, text in enumerate(section_chunks):
                if len(text) < cfg.MIN_CHUNK_CHARS:
                    continue
                chunk_id = f"{paper_id}_{section.order}_{i}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    text=text,
                    metadata={
                        "section_type": section.section_type,
                        "section_title": section.title,
                        "section_order": section.order,
                        "chunk_index_in_section": i,
                        "paper_id": paper_id,
                    },
                ))
        return chunks

    def _chunk_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if len(text) >= cfg.MIN_CHUNK_CHARS else []

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                for sep in (". ", ".\n", "? ", "! ", "\n\n"):
                    pos = text.rfind(sep, start + self.overlap, end)
                    if pos != -1:
                        end = pos + len(sep)
                        break
            chunk = text[start:end].strip()
            if len(chunk) >= cfg.MIN_CHUNK_CHARS:
                chunks.append(chunk)
            start = end - self.overlap
            if start >= len(text):
                break
        return chunks


# ── Embedding Generator ───────────────────────────────────────────────────────

class EmbeddingGenerator:
    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(cfg.EMBEDDING_MODEL)
            logger.info(f"Embedding model '{cfg.EMBEDDING_MODEL}' loaded.")
        return self._model

    def generate(self, texts: List[str]) -> List[List[float]]:
        model = self._load_model()
        return model.encode(texts, show_progress_bar=False).tolist()

    def generate_for_chunks(self, chunks: List[Chunk]):
        """
        PERF-01: Process chunks in batches to prevent OOM on large papers.
        Batches of cfg.MAX_EMBEDDING_BATCH_SIZE (default 50) instead of one giant tensor.
        """
        batch_sz = cfg.MAX_EMBEDDING_BATCH_SIZE
        results = []
        for i in range(0, len(chunks), batch_sz):
            batch = chunks[i:i + batch_sz]
            embeddings = self.generate([c.text for c in batch])
            results.extend(zip([c.chunk_id for c in batch], embeddings))
            logger.debug(f"Embedded batch {i // batch_sz + 1} ({len(batch)} chunks)")
        return results


# ── Vector Store ──────────────────────────────────────────────────────────────

class VectorStore:
    COLLECTION = "research_papers"

    def __init__(self, embedder: "EmbeddingGenerator | None" = None):
        """
        LOG-09: Accept injected EmbeddingGenerator.
        When provided, query() reuses it instead of loading a fresh model each call.
        """
        self._client = None
        self._collection = None
        self._embedder: "EmbeddingGenerator | None" = embedder

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=cfg.CHROMA_DIR)
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_chunks(self, chunks: List[Chunk], embeddings: list):
        col = self._get_collection()
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [c.metadata for c in chunks]
        embs = [e for _, e in embeddings]

        batch = 100
        for i in range(0, len(ids), batch):
            col.upsert(
                ids=ids[i:i+batch],
                documents=docs[i:i+batch],
                embeddings=embs[i:i+batch],
                metadatas=metas[i:i+batch],
            )

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        col = self._get_collection()

        # LOG-09: reuse injected embedder — no per-query model reload (was 90MB RAM spike)
        embedder = self._embedder or EmbeddingGenerator()
        query_embedding = embedder.generate([query_text])

        kwargs: Dict[str, Any] = {
            "query_embeddings": query_embedding,
            "n_results": min(n_results, col.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        results = col.query(**kwargs)
        output = []
        for i, doc in enumerate(results["documents"][0]):
            output.append({
                "content": doc,
                "score": round(1 - results["distances"][0][i], 4),
                "metadata": results["metadatas"][0][i],
                "paper_id": results["metadatas"][0][i].get("paper_id", ""),
                "section_type": results["metadatas"][0][i].get("section_type", ""),
            })
        return output

    def delete_paper(self, paper_id: str):
        col = self._get_collection()
        col.delete(where={"paper_id": paper_id})

    def get_paper_context(self, paper_id: str, max_chars: int = 4000) -> str:
        """Return a summary context string for a paper (used by chat system)."""
        col = self._get_collection()
        results = col.get(
            where={"paper_id": paper_id},
            include=["documents", "metadatas"],
            limit=20,
        )
        if not results or not results["documents"]:
            return ""

        priority = {"abstract": 0, "introduction": 1, "conclusion": 2, "results": 3}
        items = list(zip(results["documents"], results["metadatas"]))
        items.sort(key=lambda x: priority.get(x[1].get("section_type", ""), 99))

        context = ""
        for doc, _ in items:
            if len(context) + len(doc) > max_chars:
                break
            context += doc + "\n\n"
        return context.strip()


# ── Processing Pipeline ───────────────────────────────────────────────────────

class ProcessingPipeline:
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.pubmed_extractor = PubMedExtractor()
        self.chunker = TextChunker()
        self.embedder = EmbeddingGenerator()
        # LOG-09: inject shared embedder so VectorStore.query() reuses it
        self.vector_store = VectorStore(embedder=self.embedder)

    async def process_paper_async(self, file_path: str, source: str, paper_id: str) -> ExtractedPaper:
        """
        Async wrapper: runs CPU-heavy extraction + embedding in a thread pool
        so it does not block the FastAPI event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._process_sync,
            file_path,
            source,
            paper_id,
        )

    def _process_sync(self, file_path: str, source: str, paper_id: str) -> ExtractedPaper:
        """Synchronous full pipeline — runs in thread pool."""
        t0 = time.time()

        if source == "arxiv":
            paper = self.pdf_extractor.extract(file_path)
        elif source == "pubmed":
            paper = self.pubmed_extractor.extract(file_path)
        else:
            raise ValueError(f"Unknown source: {source}")

        paper.paper_id = paper_id
        logger.info(f"[{paper_id}] Extracted {len(paper.sections)} sections in {time.time()-t0:.1f}s")

        chunks = self.chunker.chunk_sections(paper_id, paper.sections)
        if not chunks:
            raise ValueError(f"[{paper_id}] No chunks produced — paper may be empty or too short.")
        logger.info(f"[{paper_id}] Produced {len(chunks)} chunks")

        t1 = time.time()
        # PERF-01: batched embedding
        embeddings = self.embedder.generate_for_chunks(chunks)
        logger.info(f"[{paper_id}] Embedded {len(chunks)} chunks in {time.time()-t1:.1f}s")

        self.vector_store.add_chunks(chunks, embeddings)
        logger.info(f"[{paper_id}] Stored in ChromaDB. Total pipeline: {time.time()-t0:.1f}s")

        paper._chunks = chunks
        return paper


# ── Singletons ────────────────────────────────────────────────────────────────

pipeline = ProcessingPipeline()
# LOG-09: single instance — shares embedder and ChromaDB client with the pipeline
vector_store = pipeline.vector_store