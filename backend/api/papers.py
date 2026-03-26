"""
papers.py — Paper ingestion endpoints and the full processing pipeline task.

Every route returns explicit success or failure — no silent failures.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from config import cfg
from database import db
from ingestion import arxiv_fetcher, pubmed_fetcher
from ingestion.local_uploader import handle_upload
from ingestion.validator import DocumentInput, ValidationError
from processing.chunker import chunk
from processing.embedder import embed
from processing.extractor import ExtractionError, extract
from processing.vector_store import VectorStore
from chat.session import create_session
from api.progress import push_paper_event, get_or_create_paper_queue

logger = logging.getLogger(__name__)

router = APIRouter()

# Shared VectorStore — created once on first use
_vector_store: VectorStore | None = None


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        from processing.embedder import embed as embed_fn
        _vector_store = VectorStore(embed_fn=embed_fn)
    return _vector_store


# ── Pydantic models ───────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    topic:     str            = Field(..., min_length=1, max_length=200)
    limit:     int            = Field(10, ge=1, le=50)
    source:    str            = Field("both")   # arxiv | pubmed | both
    date_from: Optional[str] = None             # ISO date string YYYY-MM-DD


def _paper_to_dict(paper) -> dict:
    return {
        "paper_id":       paper.paper_id,
        "source":         paper.source,
        "title":          paper.title,
        "abstract":       paper.abstract,
        "authors":        paper.authors or [],
        "url":            paper.url,
        "pipeline_stage": paper.pipeline_stage,
        "chunk_count":    paper.chunk_count,
        "error_message":  paper.error_message,
        "topic":          paper.topic,
        "created_at":     paper.created_at.isoformat() if paper.created_at else None,
        "processed_at":   paper.processed_at.isoformat() if paper.processed_at else None,
    }


# ── Full pipeline background task ─────────────────────────────────────────────

async def _run_pipeline(doc: DocumentInput) -> None:
    """
    Run the full processing pipeline for a single document.

    Stages: downloading → downloaded → processing → processed
    Pushes SSE events at each stage.
    Sets failed_download or failed_processing on error.
    Never raises — all failures are caught and recorded.
    """
    paper_id = doc.paper_id

    async def _stage(stage: str, msg: str) -> None:
        async with db.session() as sess:
            await db.set_stage(sess, paper_id, stage)
        await push_paper_event(paper_id, "progress", {"stage": stage, "message": msg})
        logger.info("Pipeline %s: %s — %s", paper_id, stage, msg)

    async def _fail(stage: str, error: str) -> None:
        async with db.session() as sess:
            await db.set_stage(sess, paper_id, stage, error=error)
            await db.log(sess, paper_id, stage, "failed", error, 0.0)
        await push_paper_event(paper_id, "done", {
            "paper_id": paper_id, "success": False,
            "stage": stage, "error": error,
        })
        logger.error("Pipeline failed %s at %s: %s", paper_id, stage, error)

    try:
        import time

        # ── Stage 1: Download ──────────────────────────────────────────────────
        await _stage("downloading", "Downloading document")
        t0 = time.monotonic()

        if doc.source in ("arxiv", "pubmed") and not doc.file_path:
            pdf_url = doc.extra_metadata.get("pdf_url", "").strip()
            if not pdf_url:
                # No PDF URL available — PubMed abstract-only path.
                file_path = ""
            else:
                dest_dir = Path(cfg.DOWNLOADS_DIR) / doc.source
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / f"{doc.paper_id}.pdf"
                try:
                    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                        response = await client.get(pdf_url)
                        response.raise_for_status()
                        dest_path.write_bytes(response.content)
                    file_path = str(dest_path)
                except httpx.HTTPStatusError as exc:
                    await _fail("failed_download", f"HTTP {exc.response.status_code} downloading PDF: {pdf_url}")
                    return

            async with db.session() as sess:
                await db.upsert_paper(sess, {
                    "paper_id":  paper_id,
                    "file_path": file_path,
                    "source":    doc.source,
                    "title":     doc.title,
                    "abstract":  doc.abstract,
                    "authors":   doc.authors,
                    "url":       doc.url,
                    "topic":     doc.topic,
                })
            doc = DocumentInput(
                paper_id=doc.paper_id, source=doc.source, title=doc.title,
                abstract=doc.abstract, authors=doc.authors, url=doc.url,
                file_path=file_path, topic=doc.topic,
                extra_metadata=doc.extra_metadata,
            )
        else:
            file_path = doc.file_path

        download_duration = time.monotonic() - t0
        async with db.session() as sess:
            await db.log(sess, paper_id, "downloading", "completed", "Downloaded", download_duration)
        await _stage("downloaded", "Download complete")

        # ── Stage 2: Process ───────────────────────────────────────────────────
        if not file_path:
            # FIX: PubMed abstract-only path — mark processed but do NOT create
            # a chat session. There are no embedded chunks to retrieve, so any
            # chat attempt would immediately fail with "No relevant content found".
            # The frontend filters on chunk_count > 0 to decide whether to show
            # a session, so these papers are stored for reference only.
            async with db.session() as sess:
                await db.set_stage(sess, paper_id, "processed")
                await db.log(
                    sess, paper_id, "processing", "completed",
                    "Abstract-only record — no PDF available for this PubMed paper. "
                    "Chat is not available.", 0.0,
                )
            await push_paper_event(paper_id, "done", {
                "paper_id":    paper_id,
                "success":     True,
                "chunk_count": 0,
                "message":     "Abstract saved — no full text available for this PubMed paper",
            })
            logger.info(
                "Pipeline complete (abstract-only): %s — no chat session created", paper_id
            )
            return

        await _stage("processing", "Extracting text")
        t1 = time.monotonic()

        try:
            extracted = await extract(doc)
        except ExtractionError as exc:
            await _fail("failed_processing", f"Extraction failed: {exc}")
            return

        chunks = chunk(extracted)
        if not chunks:
            await _fail("failed_processing", "Chunking produced no chunks — document may be empty or image-only")
            return

        await push_paper_event(paper_id, "progress", {
            "stage":   "processing",
            "message": f"Embedding {len(chunks)} chunks",
        })

        vectors = await embed(chunks)

        vs = _get_vector_store()
        await vs.add_chunks(chunks, vectors)

        process_duration = time.monotonic() - t1

        async with db.session() as sess:
            await db.upsert_paper(sess, {"paper_id": paper_id, "source": doc.source, "chunk_count": len(chunks)})
            await db.set_stage(sess, paper_id, "processed")
            await db.log(sess, paper_id, "processing", "completed",
                         f"Processed {len(chunks)} chunks", process_duration)

        # Only create a chat session when there are actual chunks to retrieve
        await create_session(paper_id, doc.topic or doc.title or paper_id, "beginner")

        await push_paper_event(paper_id, "done", {
            "paper_id":    paper_id,
            "success":     True,
            "chunk_count": len(chunks),
            "message":     f"Processing complete — {len(chunks)} chunks indexed",
        })
        logger.info("Pipeline complete: %s (%d chunks)", paper_id, len(chunks))

    except Exception as exc:
        await _fail("failed_processing", f"Unexpected error: {exc}")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/papers/search")
async def search_papers(request: SearchRequest, background_tasks: BackgroundTasks):
    """
    Search arXiv and/or PubMed and start the pipeline for each result.
    Returns the initial paper records immediately; processing happens in the background.
    """
    date_from = None
    if request.date_from:
        try:
            date_from = datetime.fromisoformat(request.date_from)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date_from format: {request.date_from!r}. Use YYYY-MM-DD.",
            )

    source = request.source.lower()
    if source not in ("arxiv", "pubmed", "both"):
        raise HTTPException(
            status_code=400,
            detail=f"source must be arxiv, pubmed, or both. Got: {source!r}",
        )

    docs: list[DocumentInput] = []

    try:
        if source in ("arxiv", "both"):
            arxiv_docs = await arxiv_fetcher.search(request.topic, request.limit, date_from)
            docs.extend(arxiv_docs)

        if source in ("pubmed", "both"):
            pubmed_docs = await pubmed_fetcher.search(request.topic, request.limit, date_from)
            docs.extend(pubmed_docs)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Search failed: {exc}")

    if not docs:
        return {"papers": [], "message": f"No papers found for topic {request.topic!r}"}

    papers = []
    async with db.session() as sess:
        for doc in docs:
            paper = await db.upsert_paper(sess, {
                "paper_id":       doc.paper_id,
                "source":         doc.source,
                "title":          doc.title,
                "abstract":       doc.abstract,
                "authors":        doc.authors,
                "url":            doc.url,
                "file_path":      doc.file_path,
                "topic":          doc.topic,
                "pipeline_stage": "pending",
            })
            papers.append(paper)

    for doc in docs:
        get_or_create_paper_queue(doc.paper_id)
        background_tasks.add_task(_run_pipeline, doc)

    return {
        "papers":  [_paper_to_dict(p) for p in papers],
        "message": f"Found {len(docs)} papers. Processing started.",
    }


@router.post("/papers/upload")
async def upload_paper(
    background_tasks: BackgroundTasks,
    file:  UploadFile = File(...),
    topic: str        = Form(...),
):
    """
    Accept a local PDF upload and start the pipeline.
    Returns the paper record immediately; processing happens in the background.
    """
    try:
        doc = await handle_upload(file, topic)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    async with db.session() as sess:
        paper = await db.upsert_paper(sess, {
            "paper_id":       doc.paper_id,
            "source":         doc.source,
            "title":          doc.title,
            "abstract":       doc.abstract,
            "authors":        doc.authors,
            "url":            doc.url,
            "file_path":      doc.file_path,
            "topic":          doc.topic,
            "pipeline_stage": "pending",
        })

    get_or_create_paper_queue(doc.paper_id)
    background_tasks.add_task(_run_pipeline, doc)

    return {
        "paper":   _paper_to_dict(paper),
        "message": "Upload accepted. Processing started.",
    }


@router.get("/papers")
async def list_papers(
    stage:  Optional[str] = None,
    source: Optional[str] = None,
    limit:  int = 50,
    offset: int = 0,
):
    """Return a filtered list of papers."""
    async with db.session() as sess:
        papers = await db.list_papers(sess, stage=stage, source=source,
                                      limit=limit, offset=offset)
    return {"papers": [_paper_to_dict(p) for p in papers]}


@router.delete("/papers/{paper_id}")
async def delete_paper(paper_id: str):
    """
    Delete a paper: remove ChromaDB vectors first, then the database record.
    Returns a clear error if the paper does not exist.
    """
    async with db.session() as sess:
        paper = await db.get_paper(sess, paper_id)

    if paper is None:
        raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id!r}")

    try:
        vs = _get_vector_store()
        vs.delete_paper(paper_id)
    except Exception as exc:
        logger.warning("ChromaDB delete failed for %s: %s", paper_id, exc)

    async with db.session() as sess:
        deleted = await db.delete_paper(sess, paper_id)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id!r}")

    return {"success": True, "paper_id": paper_id, "message": "Paper deleted successfully."}