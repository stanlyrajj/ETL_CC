"""
papers.py — Paper ingestion endpoints and the full processing pipeline task.

Two-phase search flow:
  POST /papers/search  — fetch metadata from arXiv/PubMed, return for user preview.
                         Does NOT start the pipeline. No DB writes.
  POST /papers/process — accept selected paper IDs, save to DB, start pipeline.

Every route returns explicit success or failure — no silent failures.
"""

import asyncio
import logging
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

_vector_store: VectorStore | None = None


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        from processing.embedder import embed as embed_fn
        _vector_store = VectorStore(embed_fn=embed_fn)
    return _vector_store


# ── Pydantic models ───────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    topic:     str           = Field(..., min_length=1, max_length=200)
    limit:     int           = Field(10, ge=1, le=50)
    source:    str           = Field("both")
    sort_by:   str           = Field("date")
    date_from: Optional[str] = None
    date_to:   Optional[str] = None
    category:  Optional[str] = None
    keyword:   Optional[str] = None


class ProcessRequest(BaseModel):
    paper_ids: list[str] = Field(..., min_length=1)


# ── CHANGE: _doc_to_preview in papers.py ─────────────────────────────────────
# Replace the existing _doc_to_preview function with this one.
# Only the has_pdf → has_full_text rename and its value logic changed.
# Everything else is identical to the original.

def _doc_to_preview(doc: DocumentInput) -> dict:
    return {
        "paper_id":      doc.paper_id,
        "source":        doc.source,
        "title":         doc.title,
        "abstract":      doc.abstract,
        "authors":       doc.authors,
        "url":           doc.url,
        "has_full_text": bool(
            doc.extra_metadata.get("pdf_url")
            or (doc.source == "pubmed" and doc.extra_metadata.get("pubmed_id"))
        ),
        "published":  doc.extra_metadata.get("published") or doc.extra_metadata.get("pub_date") or "",
        "journal":    doc.extra_metadata.get("journal") or "",
        "categories": doc.extra_metadata.get("categories") or [],
        "doi":        doc.extra_metadata.get("doi") or "",
    }


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


_pending_docs: dict[str, DocumentInput] = {}
_PENDING_CAP = 500


def _store_pending(doc: DocumentInput) -> None:
    if len(_pending_docs) >= _PENDING_CAP:
        evict_count = _PENDING_CAP // 4
        for key in list(_pending_docs.keys())[:evict_count]:
            del _pending_docs[key]
        logger.warning(
            "Pending docs cap reached (%d) — evicted %d oldest entries.",
            _PENDING_CAP, evict_count,
        )
    _pending_docs[doc.paper_id] = doc


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

        await _stage("downloading", "Downloading document")
        t0 = time.monotonic()

        if doc.source in ("arxiv", "pubmed") and not doc.file_path:
            pdf_url = doc.extra_metadata.get("pdf_url", "").strip()

            if pdf_url:
                dest_dir  = Path(cfg.DOWNLOADS_DIR) / doc.source
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / f"{doc.paper_id}.pdf"
                try:
                    async with httpx.AsyncClient(
                        follow_redirects=True, timeout=60.0
                    ) as client:
                        response = await client.get(pdf_url)
                        response.raise_for_status()
                        dest_path.write_bytes(response.content)
                    file_path = str(dest_path)
                    logger.info(
                        "PDF downloaded: %s → %s (%d bytes)",
                        pdf_url, dest_path, len(response.content),
                    )
                except httpx.HTTPStatusError as exc:
                    await _fail(
                        "failed_download",
                        f"HTTP {exc.response.status_code} downloading PDF: {pdf_url}",
                    )
                    return

            elif doc.source == "pubmed":
                pmid = doc.extra_metadata.get("pubmed_id", "").strip()

                if pmid:
                    await push_paper_event(paper_id, "progress", {
                        "stage":   "downloading",
                        "message": "Fetching full text from PubMed Central…",
                    })
                    bioc_dir  = Path(cfg.DOWNLOADS_DIR) / "pubmed"
                    file_path = await pubmed_fetcher.fetch_bioc_json(
                        pmid=pmid,
                        paper_id=doc.paper_id,
                        output_dir=str(bioc_dir),
                    )
                    if file_path:
                        logger.info(
                            "BioC full text fetched for %s: %s", paper_id, file_path
                        )
                    else:
                        logger.info(
                            "No full text available for %s (PMID=%s) — abstract only",
                            paper_id, pmid,
                        )
                        file_path = ""
                else:
                    logger.warning(
                        "PubMed paper %s has no pubmed_id in metadata — abstract only",
                        paper_id,
                    )
                    file_path = ""
            else:
                file_path = ""

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
            await db.log(
                sess, paper_id, "downloading", "completed", "Downloaded",
                download_duration,
            )
        await _stage("downloaded", "Download complete")

        if not file_path:
            async with db.session() as sess:
                await db.set_stage(sess, paper_id, "processed")
                await db.log(
                    sess, paper_id, "processing", "completed",
                    "Abstract-only record — no full text available.", 0.0,
                )
            await push_paper_event(paper_id, "done", {
                "paper_id":    paper_id,
                "success":     True,
                "chunk_count": 0,
                "message":     "Abstract saved — no full text available for this paper",
            })
            await create_session(
                paper_id,
                doc.topic or doc.title or paper_id,
                "beginner",
            )
            logger.info("Pipeline complete (abstract-only): %s", paper_id)
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
            await _fail(
                "failed_processing",
                "Chunking produced no chunks — document may be empty or image-only",
            )
            return

        await push_paper_event(paper_id, "progress", {
            "stage":   "processing",
            "message": f"Embedding {len(chunks)} chunks",
        })

        vectors = await embed(chunks)
        vs      = _get_vector_store()
        await vs.add_chunks(chunks, vectors)

        process_duration = time.monotonic() - t1

        async with db.session() as sess:
            await db.upsert_paper(sess, {
                "paper_id":    paper_id,
                "source":      doc.source,
                "chunk_count": len(chunks),
            })
            await db.set_stage(sess, paper_id, "processed")
            await db.log(
                sess, paper_id, "processing", "completed",
                f"Processed {len(chunks)} chunks", process_duration,
            )

        await create_session(
            paper_id,
            doc.topic or doc.title or paper_id,
            "beginner",
        )

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
async def search_papers(request: SearchRequest):
    """
    Search arXiv and/or PubMed and return paper previews for user selection.

    Does NOT start the pipeline or write to the database.
    """
    source = request.source.lower()
    if source not in ("arxiv", "pubmed", "both"):
        raise HTTPException(
            status_code=400,
            detail=f"source must be arxiv, pubmed, or both. Got: {source!r}",
        )

    if request.sort_by not in ("date", "relevance"):
        raise HTTPException(
            status_code=400,
            detail=f"sort_by must be 'date' or 'relevance'. Got: {request.sort_by!r}",
        )

    date_from = date_to = None
    if request.date_from:
        try:
            date_from = datetime.fromisoformat(request.date_from)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date_from: {request.date_from!r}. Use YYYY-MM-DD.",
            )
    if request.date_to:
        try:
            date_to = datetime.fromisoformat(request.date_to)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date_to: {request.date_to!r}. Use YYYY-MM-DD.",
            )

    docs: list[DocumentInput] = []
    errors: list[str] = []

    try:
        if source in ("arxiv", "both"):
            arxiv_docs = await arxiv_fetcher.search(
                topic=request.topic,
                limit=request.limit,
                date_from=date_from,
                date_to=date_to,
                sort_by=request.sort_by,
                category=request.category or None,
                keyword=request.keyword or None,
            )
            docs.extend(arxiv_docs)
    except Exception as exc:
        # arXiv failure is non-fatal when source=both — PubMed may still work
        if source == "arxiv":
            raise HTTPException(
                status_code=502,
                detail=f"arXiv search failed: {exc}",
            )
        errors.append(f"arXiv: {exc}")
        logger.warning("arXiv search failed (continuing with PubMed): %s", exc)

    try:
        if source in ("pubmed", "both"):
            pubmed_docs = await pubmed_fetcher.search(
                topic=request.topic,
                limit=request.limit,
                date_from=date_from,
                date_to=date_to,
                sort_by=request.sort_by,
                keyword=request.keyword or None,
            )
            docs.extend(pubmed_docs)
    except Exception as exc:
        if source == "pubmed":
            raise HTTPException(
                status_code=502,
                detail=f"PubMed search failed: {exc}",
            )
        errors.append(f"PubMed: {exc}")
        logger.warning("PubMed search failed (continuing with arXiv results): %s", exc)

    # If source=both and BOTH failed, raise now
    if source == "both" and not docs and errors:
        raise HTTPException(
            status_code=502,
            detail=(
                "Both arXiv and PubMed searches failed. "
                "Check your internet connection. "
                f"Errors: {'; '.join(errors)}"
            ),
        )

    if not docs:
        message = f"No papers found for topic {request.topic!r}"
        if errors:
            message += f" (some sources failed: {'; '.join(errors)})"
        return {"papers": [], "message": message}

    for doc in docs:
        _store_pending(doc)

    message = f"Found {len(docs)} papers. Select which ones to process."
    if errors:
        message += f" (note: {'; '.join(errors)})"

    return {"papers": [_doc_to_preview(doc) for doc in docs], "message": message}


@router.post("/papers/process")
async def process_papers(request: ProcessRequest, background_tasks: BackgroundTasks):
    """Start the pipeline for a user-selected subset of search results."""
    if not request.paper_ids:
        raise HTTPException(status_code=400, detail="No paper IDs provided.")

    docs: list[DocumentInput] = []
    missing: list[str] = []
    for paper_id in request.paper_ids:
        doc = _pending_docs.get(paper_id)
        if doc is None:
            missing.append(paper_id)
        else:
            docs.append(doc)

    if missing:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Paper ID(s) not found in search results: {missing}. "
                "Run POST /papers/search first."
            ),
        )

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

    for paper_id in request.paper_ids:
        _pending_docs.pop(paper_id, None)

    return {
        "papers":  [_paper_to_dict(p) for p in papers],
        "message": f"Processing started for {len(docs)} paper(s).",
    }


@router.post("/papers/upload")
async def upload_paper(
    background_tasks: BackgroundTasks,
    file:  UploadFile = File(...),
    topic: str        = Form(...),
):
    """Accept a local PDF upload and start the pipeline."""
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
        papers = await db.list_papers(
            sess, stage=stage, source=source, limit=limit, offset=offset
        )
    return {"papers": [_paper_to_dict(p) for p in papers]}


@router.delete("/papers/{paper_id}")
async def delete_paper(paper_id: str):
    """Delete a paper and all related data."""
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

    return {
        "success":  True,
        "paper_id": paper_id,
        "message":  "Paper deleted successfully.",
    }