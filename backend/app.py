"""
app.py — Single unified FastAPI application.

Fixes applied (in batch order):
  Batch 1 — CRITICAL:
    SEC-01: Prompt injection — paper context sanitized + XML-delimited in gemini_client.py
    SEC-03: Per-IP rate limiting via slowapi (search=5/min, chat=30/min, generate=3/min)
            + asyncio.Semaphore(5) caps concurrent pipeline tasks
    LOG-01: PubMed download switched from efetch (returns XML) to BioC JSON endpoint
    LOG-02: Idempotency guard expanded to all active stages + _active_pipelines set
    LOG-03: Orphaned pipeline recovery on startup (_recover_orphaned_pipelines)
    LOG-04: ChromaDB failure returns 503 instead of silently hallucinating
  Batch 2 — HIGH:
    SEC-02: Log filter redacts GEMINI_API_KEY from all log records
    SEC-04: _safe_filename() + _assert_within_dir() prevent path traversal
    LOG-05: delete_paper() — ChromaDB deleted before PostgreSQL (no orphaned vectors)
    LOG-06: RateLimiter uses asyncio.Lock (concurrency-safe, no TOCTOU race)
    SEC-05: Optional X-API-Key authentication via _auth dependency
  Batch 3 — HIGH:
    LOG-08: SSE generators wrapped in try/finally — queues always cleaned on disconnect
    LOG-10: gemini calls are all awaited (async in gemini_client.py)
    PERF-03: _search_pubmed() uses async httpx instead of blocking requests.get()
  Batch 4 — MEDIUM:
    PERF-05: arxiv.Search iterator runs in thread pool executor (sync HTTP, blocks loop)
    ARCH-02: Playwright startup check; carousel returns 503 if Chromium missing
"""

import asyncio
import json
import logging
import os
import re as _re
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request

from config import cfg
from database import db
from gemini_client import get_gemini
from processor import pipeline, vector_store

# ── Logging + SEC-02 key-redaction filter ─────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


class _RedactSecretsFilter(logging.Filter):
    """Scrub GEMINI_API_KEY from every log record before emission."""
    def filter(self, record: logging.LogRecord) -> bool:
        secret = cfg.GEMINI_API_KEY
        if secret and len(secret) > 8:
            msg = str(record.msg)
            if secret in msg:
                record.msg = msg.replace(secret, "[REDACTED]")
            if record.args:
                args = record.args if isinstance(record.args, tuple) else (record.args,)
                record.args = tuple(
                    str(a).replace(secret, "[REDACTED]") if isinstance(a, str) else a
                    for a in args
                )
        return True


logging.getLogger().addFilter(_RedactSecretsFilter())
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ResearchRAG API",
    description="Research paper discovery, RAG, teaching chat, and content generation",
    version="3.0.0",
)

# SEC-03: per-IP rate limiting
_limiter = Limiter(key_func=get_remote_address)
app.state.limiter = _limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# SEC-03: cap concurrent background pipeline tasks to prevent resource exhaustion
_pipeline_sem = asyncio.Semaphore(5)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SEC-05: optional API key auth
_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _auth(key: Optional[str] = Depends(_key_header)) -> None:
    """Reject requests without valid API key when API_SECRET_KEY is configured."""
    if cfg.API_SECRET_KEY and key != cfg.API_SECRET_KEY:
        raise HTTPException(401, "Unauthorized")


# ── Rate limiter (LOG-06: asyncio.Lock makes check-sleep-update atomic) ────────

class RateLimiter:
    """Concurrency-safe async rate limiter. asyncio.Lock serializes check+sleep+update."""

    def __init__(self, min_interval: float):
        self._min = min_interval
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            elapsed = time.time() - self._last
            if elapsed < self._min:
                await asyncio.sleep(self._min - elapsed)
            self._last = time.time()


arxiv_limiter = RateLimiter(cfg.ARXIV_RATE_LIMIT_SECONDS)
pubmed_limiter = RateLimiter(cfg.PUBMED_RATE_LIMIT_SECONDS)

# SEC-04: filesystem safety helpers
def _safe_filename(paper_id: str, max_len: int = 100) -> str:
    """Sanitize paper_id for filesystem use. Strips everything except [a-zA-Z0-9._-]."""
    safe = _re.sub(r'[^a-zA-Z0-9._-]', '_', str(paper_id))[:max_len]
    if not safe or safe in (".", ".."):
        raise ValueError(f"Unsafe paper_id for filesystem: {paper_id!r}")
    return safe


def _assert_within_dir(path: Path, base_dir: Path) -> None:
    """Raise ValueError if resolved path would escape base_dir."""
    if not str(path.resolve()).startswith(str(base_dir.resolve()) + "/"):
        raise ValueError(f"Path traversal detected: {path} escapes {base_dir}")


# ── In-memory SSE event queues ─────────────────────────────────────────────────

_sse_queues: Dict[str, asyncio.Queue] = {}
# LOG-02: track actively-running pipeline tasks (in-memory dedup)
_active_pipelines: set = set()


def _push_event(paper_id: str, event: str, data: Dict):
    q = _sse_queues.get(paper_id)
    if q:
        try:
            q.put_nowait({"event": event, "data": data})
        except asyncio.QueueFull:
            pass


# ── LOG-03: Orphaned pipeline recovery ────────────────────────────────────────

async def _recover_orphaned_pipelines() -> None:
    """On startup: reset and re-queue any papers stuck mid-pipeline from a prior crash."""
    orphan_stages = ["downloading", "downloaded", "processing"]
    recovered = 0
    for stage in orphan_stages:
        async with db.session() as sess:
            orphans = await db.list_papers_by_stage(sess, stage=stage, limit=200)
        for paper in orphans:
            logger.warning(f"[recovery] {paper.paper_id} stuck at '{stage}' — resetting to pending")
            async with db.session() as sess:
                await db.set_stage(sess, paper.paper_id, "pending")
            paper_data = {
                "paper_id": paper.paper_id,
                "source":   paper.source,
                "title":    paper.title or "",
                "topic":    (paper.extra_metadata or {}).get("topic", paper.title or paper.paper_id),
                **(paper.extra_metadata or {}),
            }
            _sse_queues[paper.paper_id] = asyncio.Queue(maxsize=50)
            _active_pipelines.add(paper.paper_id)
            asyncio.create_task(_full_pipeline_task(paper_data, paper.paper_id))
            recovered += 1
    logger.info(
        f"[recovery] {'No orphaned pipelines.' if not recovered else f'Restarted {recovered} pipeline(s).'}"
    )


# ── ARCH-02: Playwright startup check ─────────────────────────────────────────

_playwright_ok: bool = False


async def _check_playwright() -> bool:
    global _playwright_ok
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            b = await pw.chromium.launch()
            await b.close()
        _playwright_ok = True
        logger.info("[startup] Playwright/Chromium: OK")
    except Exception as exc:
        _playwright_ok = False
        logger.warning(f"[startup] Carousel UNAVAILABLE — run: playwright install chromium  ({exc})")
    return _playwright_ok


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    cfg.validate()
    await db.init()
    await _recover_orphaned_pipelines()  # LOG-03
    await _check_playwright()            # ARCH-02
    logger.info("ResearchRAG API ready.")


@app.on_event("shutdown")
async def shutdown():
    await db.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAPERS — Search, Download, List, Delete
# ══════════════════════════════════════════════════════════════════════════════

class SearchRequest(BaseModel):
    topic: str
    limit: int = 5
    source: str = "both"
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    sort_by: str = "relevance"

    @validator("topic")
    def topic_not_empty(cls, v):
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Topic must be at least 3 characters.")
        if len(v) > 500:
            raise ValueError("Topic must be under 500 characters.")
        return v

    @validator("limit")
    def limit_range(cls, v):
        return max(1, min(v, 20))


async def _search_arxiv(topic: str, limit: int, date_from: str = None) -> List[Dict]:
    await arxiv_limiter.wait()
    try:
        import arxiv
        query = topic
        if date_from:
            query += f" AND submittedDate:[{date_from.replace('-', '')} TO 99991231]"

        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        # PERF-05: arxiv library uses blocking requests — run in executor
        loop = asyncio.get_event_loop()
        _arxiv_results = await loop.run_in_executor(None, lambda: list(search.results()))
        results = []
        for r in _arxiv_results:
            results.append({
                "paper_id": r.entry_id.split("/")[-1],
                "source": "arxiv",
                "title": r.title,
                "authors": [str(a) for a in r.authors[:5]],
                "abstract": r.summary[:500],
                "url": r.entry_id,
                "published_date": r.published.isoformat() if r.published else None,
                "pdf_url": r.pdf_url,
            })
        return results
    except Exception as exc:
        logger.error(f"arXiv search failed: {exc}")
        return []


async def _search_pubmed(topic: str, limit: int, date_from: str = None) -> List[Dict]:
    await pubmed_limiter.wait()
    try:
        date_filter = ""
        if date_from:
            date_filter = f" AND {date_from}[PDAT]:3000[PDAT]"

        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pmc",
            "term": f"{topic} open access[filter]{date_filter}",
            "retmax": limit,
            "retmode": "json",
            "sort": "pub+date",
        }

        # PERF-03: async httpx replaces blocking requests.get() (was blocking event loop)
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(search_url, params=params)
            resp.raise_for_status()
            ids = resp.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []

            await pubmed_limiter.wait()

            summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            sum_params = {"db": "pmc", "id": ",".join(ids), "retmode": "json"}
            sum_resp = await client.get(summary_url, params=sum_params)
            sum_resp.raise_for_status()

        summaries = sum_resp.json().get("result", {})
        results = []
        for uid in ids:
            s = summaries.get(uid, {})
            if not s or uid == "uids":
                continue
            results.append({
                "paper_id": f"PMC{uid}",
                "source": "pubmed",
                "title": s.get("title", "Unknown"),
                "authors": [a.get("name", "") for a in s.get("authors", [])[:5]],
                "abstract": "",
                "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{uid}/",
                "published_date": s.get("pubdate", ""),
                "pmc_id": uid,
            })
        return results
    except Exception as exc:
        logger.error(f"PubMed search failed: {exc}")
        return []


async def _download_arxiv(paper_data: Dict) -> str:
    """Download arXiv PDF. Returns local file path."""
    await arxiv_limiter.wait()
    paper_id = paper_data["paper_id"]
    out_dir = Path(cfg.DOWNLOADS_DIR) / "arxiv"
    out_dir.mkdir(parents=True, exist_ok=True)

    # SEC-04: sanitize paper_id before filesystem use
    _sfid = _safe_filename(paper_id)
    _out_path_obj = out_dir / f"{_sfid}.pdf"
    _assert_within_dir(_out_path_obj, out_dir)
    out_path = str(_out_path_obj)

    if Path(out_path).exists():
        logger.info(f"[{paper_id}] PDF already cached.")
        return out_path

    pdf_url = paper_data.get("pdf_url") or f"https://arxiv.org/pdf/{paper_id}.pdf"

    for attempt in range(1, cfg.MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
                resp = await client.get(pdf_url, headers={"User-Agent": "ResearchRAG/3.0"})
                resp.raise_for_status()
                with open(out_path, "wb") as f:
                    f.write(resp.content)
            logger.info(f"[{paper_id}] Downloaded PDF ({len(resp.content)//1024}KB)")
            return out_path
        except Exception as exc:
            if attempt < cfg.MAX_RETRIES:
                wait = cfg.RETRY_BACKOFF_BASE ** (attempt - 1)
                logger.warning(f"[{paper_id}] Download attempt {attempt} failed: {exc}. Retry in {wait}s")
                await asyncio.sleep(wait)
            else:
                raise RuntimeError(f"arXiv download failed after {cfg.MAX_RETRIES} attempts: {exc}")


async def _download_pubmed(paper_data: Dict) -> str:
    """
    Download PubMed BioC JSON. Returns local file path.
    LOG-01: Uses BioC JSON endpoint instead of efetch (which returns XML, not JSON).
    """
    await pubmed_limiter.wait()
    paper_id = paper_data["paper_id"]
    pmc_id = paper_data.get("pmc_id", paper_id.replace("PMC", ""))
    out_dir = Path(cfg.DOWNLOADS_DIR) / "pubmed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # SEC-04: sanitize paper_id before filesystem use
    _sfid = _safe_filename(paper_id)
    _out_path_obj = out_dir / f"{_sfid}.json"
    _assert_within_dir(_out_path_obj, out_dir)
    out_path = str(_out_path_obj)

    if Path(out_path).exists():
        return out_path

    # LOG-01: BioC JSON endpoint — efetch returns XML and was never parseable as JSON
    bioc_url = f"https://www.ncbi.nlm.nih.gov/research/bioxiv/PMC{pmc_id}?format=json"

    for attempt in range(1, cfg.MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(bioc_url)
                resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "json" not in ct and not resp.text.strip().startswith("{"):
                raise ValueError(
                    f"BioC API returned non-JSON (Content-Type: {ct}). "
                    f"PMC{pmc_id} may not be open-access."
                )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(resp.text)
            logger.info(f"[{paper_id}] BioC JSON downloaded ({len(resp.content)//1024}KB)")
            return out_path
        except Exception as exc:
            if attempt < cfg.MAX_RETRIES:
                wait = cfg.RETRY_BACKOFF_BASE ** (attempt - 1)
                logger.warning(f"[{paper_id}] PubMed attempt {attempt} failed: {exc}. Retry in {wait}s")
                await asyncio.sleep(wait)
            else:
                raise RuntimeError(f"PubMed BioC download failed after {cfg.MAX_RETRIES} attempts: {exc}")


async def _full_pipeline_task(paper_data: Dict, paper_id: str):
    """
    End-to-end background task:
    download → extract → chunk → embed → store → create chat session
    Pushes SSE events at each stage.
    """
    source = paper_data["source"]

    async def push(stage: str, status: str, msg: str = ""):
        _push_event(paper_id, "progress", {"stage": stage, "status": status, "message": msg})

    # ── Stage 1: Download ──
    await push("download", "started", "Downloading paper…")
    async with db.session() as session:
        await db.set_stage(session, paper_id, "downloading")

    try:
        if source == "arxiv":
            file_path = await _download_arxiv(paper_data)
        else:
            file_path = await _download_pubmed(paper_data)

        async with db.session() as session:
            await db.set_stage(session, paper_id, "downloaded")
            await db.log(session, paper_id, "download", "completed")
        await push("download", "completed", "Download complete.")
    except Exception as exc:
        async with db.session() as session:
            await db.set_stage(session, paper_id, "failed_download", str(exc))
            await db.log(session, paper_id, "download", "failed", str(exc))
        await push("download", "failed", str(exc))
        _active_pipelines.discard(paper_id)  # LOG-02: release on failure
        _push_event(paper_id, "done", {"success": False, "error": str(exc)})
        return

    # ── Stage 2: Process (extract + chunk + embed) ──
    await push("processing", "started", "Extracting and embedding paper…")
    async with db.session() as session:
        await db.set_stage(session, paper_id, "processing")

    t0 = time.time()
    try:
        extracted = await pipeline.process_paper_async(file_path, source, paper_id)
        duration = round(time.time() - t0, 2)

        sections_data = [
            {
                "section_type": s.section_type,
                "section_title": s.title,
                "content": s.content,
                "section_order": s.order,
                "word_count": len(s.content.split()),
            }
            for s in extracted.sections
        ]
        chunks_data = [
            {
                "chunk_index": i,
                "section_type": c.metadata.get("section_type"),
                "content": c.text,
                "embedding_id": c.chunk_id,
                "char_count": len(c.text),
                "chunk_metadata": c.metadata,
            }
            for i, c in enumerate(extracted._chunks)
        ]

        async with db.session() as session:
            paper = await db.get_paper(session, paper_id)
            updates = {}
            if not paper.title and extracted.title:
                updates["title"] = extracted.title
            if not paper.abstract and extracted.abstract:
                updates["abstract"] = extracted.abstract
            if updates:
                from sqlalchemy import update as sa_update
                from database import Paper as PaperModel
                await session.execute(
                    sa_update(PaperModel)
                    .where(PaperModel.paper_id == paper_id)
                    .values(**updates, file_path=file_path)
                )
            await db.save_sections(session, paper_id, sections_data)
            await db.save_chunks(session, paper_id, chunks_data)
            await db.set_stage(session, paper_id, "processed")
            await db.log(session, paper_id, "processing", "completed",
                        f"{len(sections_data)} sections, {len(chunks_data)} chunks", duration)

        await push("processing", "completed",
                   f"Processed: {len(chunks_data)} chunks in {duration}s")

    except Exception as exc:
        async with db.session() as session:
            await db.set_stage(session, paper_id, "failed_processing", str(exc))
            await db.log(session, paper_id, "processing", "failed", str(exc))
        await push("processing", "failed", str(exc))
        _active_pipelines.discard(paper_id)  # LOG-02: release on failure
        _push_event(paper_id, "done", {"success": False, "error": str(exc)})
        return

    # ── Stage 3: Auto-create chat session ──
    session_id = str(uuid.uuid4())
    topic = paper_data.get("topic", paper_data.get("title", paper_id))
    try:
        async with db.session() as session:
            await db.create_session(session, session_id, paper_id, topic)
        await push("chat", "ready", f"Chat session created: {session_id}")
    except Exception as exc:
        logger.warning(f"[{paper_id}] Chat session creation failed: {exc}")

    _active_pipelines.discard(paper_id)  # LOG-02: release on success
    _push_event(paper_id, "done", {
        "success": True,
        "paper_id": paper_id,
        "session_id": session_id,
        "chunk_count": len(chunks_data),
    })


@app.post("/api/papers/search", dependencies=[Depends(_auth)])
@_limiter.limit("5/minute")
async def search_papers(request: Request, req: SearchRequest, background_tasks: BackgroundTasks):
    """
    Search for papers and immediately begin download + processing pipeline.
    SEC-03: 5 requests/minute per IP.
    SEC-05: requires X-API-Key if API_SECRET_KEY is configured.
    """
    results = []

    if req.source in ("arxiv", "both"):
        arxiv_results = await _search_arxiv(req.topic, req.limit, req.date_from)
        results.extend(arxiv_results)

    if req.source in ("pubmed", "both"):
        pubmed_results = await _search_pubmed(req.topic, req.limit, req.date_from)
        results.extend(pubmed_results)

    if not results:
        return {"papers": [], "message": "No papers found. Try a different topic or source."}

    queued = []
    for paper_data in results:
        pid = paper_data["paper_id"]
        paper_data["topic"] = req.topic

        async with db.session() as session:
            existing = await db.get_paper(session, pid)
            # LOG-02: skip if already active or in any non-failed stage
            _ACTIVE_STAGES = {"processed", "downloading", "downloaded", "processing"}
            if pid in _active_pipelines or (existing and existing.pipeline_stage in _ACTIVE_STAGES):
                stage = existing.pipeline_stage if existing else "active"
                paper_data["pipeline_stage"] = stage
                queued.append({**paper_data, "pipeline_stage": stage, "cached": True})
                continue
            await db.upsert_paper(session, {
                "paper_id": pid,
                "source": paper_data["source"],
                "title": paper_data.get("title"),
                "authors": paper_data.get("authors", []),
                "abstract": paper_data.get("abstract", ""),
                "url": paper_data.get("url"),
                "extra_metadata": {k: v for k, v in paper_data.items()
                                   if k not in ("paper_id", "source", "title",
                                                "authors", "abstract", "url")},
            })

        _sse_queues[pid] = asyncio.Queue(maxsize=50)
        _active_pipelines.add(pid)  # LOG-02: register before spawning

        # SEC-03: pipeline task guarded by concurrency semaphore
        async def _guarded_pipeline(pd, p_id):
            async with _pipeline_sem:
                await _full_pipeline_task(pd, p_id)

        background_tasks.add_task(_guarded_pipeline, paper_data, pid)
        queued.append({**paper_data, "pipeline_stage": "pending"})

    return {"papers": queued, "count": len(queued), "topic": req.topic}


@app.get("/api/papers/{paper_id}/progress")
async def paper_progress(paper_id: str):
    """
    Server-Sent Events stream for pipeline progress.
    LOG-08: try/finally guarantees queue cleanup even when client disconnects early.
    """
    if paper_id not in _sse_queues:
        async with db.session() as session:
            paper = await db.get_paper(session, paper_id)
        if paper and paper.pipeline_stage == "processed":
            async def _already_done():
                yield f"event: done\ndata: {json.dumps({'success': True, 'paper_id': paper_id})}\n\n"
            return StreamingResponse(_already_done(), media_type="text/event-stream")
        raise HTTPException(404, "No active pipeline for this paper.")

    q = _sse_queues[paper_id]

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=30.0)
                    data_str = json.dumps(item["data"])
                    yield f"event: {item['event']}\ndata: {data_str}\n\n"
                    if item["event"] == "done":
                        break
                except asyncio.TimeoutError:
                    yield "event: heartbeat\ndata: {}\n\n"
        finally:
            # LOG-08: always cleanup — handles client disconnect AND normal exit
            _sse_queues.pop(paper_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/papers")
async def list_papers(
    stage: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0),
):
    async with db.session() as session:
        papers = await db.list_papers(session, stage=stage, source=source, limit=limit, offset=offset)
    return {
        "count": len(papers),
        "papers": [
            {
                "paper_id": p.paper_id,
                "source": p.source,
                "title": p.title,
                "authors": p.authors,
                "abstract": (p.abstract or "")[:200],
                "pipeline_stage": p.pipeline_stage,
                "chunk_count": p.chunk_count,
                "section_count": p.section_count,
                "processed_at": p.processed_at.isoformat() if p.processed_at else None,
                "download_error": p.download_error,
                "processing_error": p.processing_error,
            }
            for p in papers
        ],
    }


@app.delete("/api/papers/{paper_id}")
async def delete_paper(paper_id: str):
    """
    LOG-05: ChromaDB deleted FIRST. If it fails, abort and return 500 so the
    PostgreSQL record is preserved — prevents permanently orphaned vectors.
    """
    async with db.session() as session:
        paper = await db.get_paper(session, paper_id)
    if not paper:
        raise HTTPException(404, "Paper not found.")

    # Step 1: ChromaDB first — if it fails we abort before touching PostgreSQL
    try:
        vector_store.delete_paper(paper_id)
    except Exception as e:
        logger.error(f"ChromaDB delete failed for {paper_id}: {e}")
        raise HTTPException(
            500,
            f"Vector cleanup failed — delete aborted to prevent orphaned vectors: {e}"
        )

    # Step 2: PostgreSQL — only reaches here if ChromaDB succeeded
    async with db.session() as session:
        deleted = await db.delete_paper(session, paper_id)
    return {"deleted": deleted, "paper_id": paper_id}


# ══════════════════════════════════════════════════════════════════════════════
# CHAT — Multi-session teaching chat with memory
# ══════════════════════════════════════════════════════════════════════════════

class MessageRequest(BaseModel):
    message: str
    level: Optional[str] = None

    @validator("message")
    def msg_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty.")
        if len(v) > 2000:
            raise ValueError("Message too long (max 2000 chars).")
        return v


@app.get("/api/chat/sessions", dependencies=[Depends(_auth)])
async def list_sessions():
    """Return all chat sessions for the sidebar. SEC-05: protected endpoint."""
    async with db.session() as session:
        sessions = await db.list_sessions(session)
    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "paper_id": s.paper_id,
                "topic": s.topic,
                "title": s.title,
                "level": s.level,
                "last_active_at": s.last_active_at.isoformat(),
                "created_at": s.created_at.isoformat(),
            }
            for s in sessions
        ]
    }


@app.get("/api/chat/sessions/{session_id}", dependencies=[Depends(_auth)])
async def get_session(session_id: str):
    """Return session info + full message history. SEC-05: protected endpoint."""
    async with db.session() as session:
        sess = await db.get_session(session, session_id)
        if not sess:
            raise HTTPException(404, "Session not found.")
        messages = await db.get_messages(session, session_id)
        paper = await db.get_paper(session, sess.paper_id)

    return {
        "session_id": sess.session_id,
        "paper_id": sess.paper_id,
        "topic": sess.topic,
        "level": sess.level,
        "paper_title": paper.title if paper else None,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "level": m.level,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ],
    }


@app.post("/api/chat/sessions/{session_id}/message")
@_limiter.limit("30/minute")
async def send_message(request: Request, session_id: str, req: MessageRequest):
    """
    Send a message and get a teaching response from Gemini.
    SEC-03: 30 requests/minute per IP.
    LOG-04: ChromaDB failure returns 503 instead of silently hallucinating.
    """
    async with db.session() as session:
        sess = await db.get_session(session, session_id)
        if not sess:
            raise HTTPException(404, "Session not found.")

        paper = await db.get_paper(session, sess.paper_id)
        if not paper:
            raise HTTPException(404, "Paper not found.")
        if paper.pipeline_stage != "processed":
            raise HTTPException(400, "Paper is not yet fully processed.")

        messages = await db.get_messages(session, session_id)

        effective_level = req.level or sess.level
        if req.level and req.level != sess.level:
            await db.update_session_level(session, session_id, req.level)

    # LOG-04: raise 503 on ChromaDB failure — never silently hallucinate
    try:
        context_results = vector_store.query(
            query_text=req.message,
            n_results=5,
            filters={"paper_id": paper.paper_id},
        )
        paper_context = "\n\n".join(r["content"] for r in context_results)
        if not paper_context:
            paper_context = vector_store.get_paper_context(paper.paper_id)
    except Exception as ctx_exc:
        logger.error(f"ChromaDB query failed for session {session_id}: {ctx_exc}")
        raise HTTPException(
            503,
            "Paper context retrieval temporarily unavailable. Please try again."
        )

    history = [{"role": m.role, "content": m.content} for m in messages]

    try:
        gemini = get_gemini()
        authors_str = ", ".join(paper.authors or []) or "Unknown authors"
        # LOG-10: chat_response is now async — must await
        response_text = await gemini.chat_response(
            paper_context=paper_context,
            paper_title=paper.title or paper.paper_id,
            paper_authors=authors_str,
            message_history=history,
            user_message=req.message,
            level=effective_level,
        )
    except Exception as exc:
        logger.error(f"Gemini chat failed: {exc}")
        raise HTTPException(503, "AI response failed. Please try again.")

    async with db.session() as session:
        await db.add_message(session, session_id, "user", req.message, effective_level)
        await db.add_message(session, session_id, "assistant", response_text, effective_level)

    return {
        "session_id": session_id,
        "role": "assistant",
        "content": response_text,
        "level": effective_level,
    }


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE — Social content + Carousel
# ══════════════════════════════════════════════════════════════════════════════

class GenerateRequest(BaseModel):
    paper_id: str
    platform: str
    content_style: str = "educational"
    carousel_style: Optional[Dict[str, Any]] = None

    @validator("platform")
    def valid_platform(cls, v):
        if v not in ("twitter", "linkedin", "carousel"):
            raise ValueError("platform must be twitter, linkedin, or carousel")
        return v


@app.post("/api/generate")
@_limiter.limit("3/minute")
async def generate_content(request: Request, req: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Generate social content for a processed paper.
    SEC-03: 3 requests/minute per IP.
    ARCH-02: returns 503 if Playwright not available and carousel requested.
    """
    async with db.session() as session:
        paper = await db.get_paper(session, req.paper_id)
        if not paper:
            raise HTTPException(404, "Paper not found.")
        if paper.pipeline_stage != "processed":
            raise HTTPException(400, "Paper must be fully processed before generating content.")
        sections = await db.get_sections(session, req.paper_id)

    # ARCH-02: guard carousel if Playwright is unavailable
    if req.platform == "carousel" and not _playwright_ok:
        raise HTTPException(503, "Carousel rendering unavailable. Run: playwright install chromium")

    gen_id = str(uuid.uuid4())[:8]
    key = f"gen_{req.paper_id}_{req.platform}"
    _sse_queues[key] = asyncio.Queue(maxsize=20)

    background_tasks.add_task(
        _generate_task, paper, sections, req.platform,
        req.content_style, req.carousel_style, key
    )
    return {"status": "queued", "gen_id": gen_id, "queue_key": key}


async def _generate_task(paper, sections, platform, style, carousel_style, queue_key):
    def push(status: str, data: Dict):
        q = _sse_queues.get(queue_key)
        if q:
            try:
                q.put_nowait({"event": status, "data": data})
            except asyncio.QueueFull:
                pass

    push("started", {"message": f"Generating {platform} content…"})

    try:
        priority = ["abstract", "conclusion", "results", "discussion", "introduction"]
        snippets = []
        for ptype in priority:
            for s in sections:
                if ptype in s.section_type.lower() and len(snippets) < 3:
                    snippets.append(s.content[:400])
        key_content = "\n\n".join(snippets)

        authors_str = ", ".join(paper.authors or []) or "Unknown authors"
        gemini = get_gemini()

        if platform == "twitter":
            # LOG-10: async gemini methods — must await
            result = await gemini.generate_twitter_thread(
                paper.title or "", authors_str,
                paper.abstract or "", key_content,
            )
            content_str = json.dumps(result)
            hashtags = result.get("hashtags", [])

        elif platform == "linkedin":
            result = await gemini.generate_linkedin_post(
                paper.title or "", authors_str,
                paper.abstract or "", key_content,
            )
            content_str = json.dumps(result)
            hashtags = result.get("hashtags", [])

        elif platform == "carousel":
            result = await gemini.generate_carousel_content(
                paper.title or "", authors_str,
                paper.abstract or "", key_content,
            )
            from carousel_renderer import render_carousel
            style_obj = carousel_style or {}
            pdf_path, png_paths = await render_carousel(
                paper_id=paper.paper_id,
                slides=result.get("slides", []),
                style=style_obj,
            )
            result["pdf_path"] = pdf_path
            result["png_paths"] = png_paths
            content_str = json.dumps(result)
            hashtags = result.get("hashtags", [])

        async with db.session() as session:
            saved = await db.save_social(
                session, paper.paper_id, platform,
                platform, content_str, hashtags, {"style": style},
            )

        push("completed", {
            "content_id": saved.id,
            "platform": platform,
            "content": json.loads(content_str),
            "hashtags": hashtags,
        })

    except Exception as exc:
        logger.error(f"Generate task failed: {exc}")
        push("failed", {"error": str(exc)})
    finally:
        q = _sse_queues.get(queue_key)
        if q:
            q.put_nowait({"event": "done", "data": {}})


@app.get("/api/generate/{queue_key}/progress")
async def generate_progress(queue_key: str):
    """
    SSE stream for content generation progress.
    LOG-08: try/finally guarantees queue cleanup on client disconnect.
    """
    if queue_key not in _sse_queues:
        raise HTTPException(404, "No active generation for this key.")

    q = _sse_queues[queue_key]

    async def stream():
        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=60.0)
                    yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"
                    if item["event"] == "done":
                        break
                except asyncio.TimeoutError:
                    yield "event: heartbeat\ndata: {}\n\n"
        finally:
            # LOG-08: guaranteed cleanup — handles disconnect AND normal exit
            _sse_queues.pop(queue_key, None)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/generate/history/{paper_id}")
async def generation_history(paper_id: str, platform: Optional[str] = None):
    async with db.session() as session:
        items = await db.list_social(session, paper_id=paper_id, platform=platform)
    return {
        "count": len(items),
        "items": [
            {
                "id": i.id,
                "platform": i.platform,
                "content_type": i.content_type,
                "status": i.status,
                "hashtags": i.hashtags,
                "created_at": i.created_at.isoformat(),
            }
            for i in items
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# QUERY — RAG semantic search
# ══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query: str
    n_results: int = 5
    paper_id: Optional[str] = None

    @validator("query")
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty.")
        return v.strip()


@app.post("/api/query")
async def query_papers(req: QueryRequest):
    t0 = time.time()
    filters = {"paper_id": req.paper_id} if req.paper_id else None
    try:
        results = vector_store.query(req.query, req.n_results, filters)
    except Exception as exc:
        raise HTTPException(503, f"Search failed: {exc}")
    return {
        "query": req.query,
        "result_count": len(results),
        "duration_ms": int((time.time() - t0) * 1000),
        "results": results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "3.0.0",
        "playwright": _playwright_ok,
    }