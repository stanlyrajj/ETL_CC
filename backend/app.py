"""
app.py — Single unified FastAPI application.

Replaces: main.py + main_part2.py (two separate apps → one).

Routers:
  /api/papers   — search, download, list, delete
  /api/process  — SSE progress stream
  /api/chat     — multi-session teaching chat
  /api/generate — social content (Twitter, LinkedIn, Carousel)
  /api/query    — RAG semantic search
  /health       — health check

Run:
    uvicorn app:app --reload --port 8000
"""

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, validator

from config import cfg
from database import db
from gemini_client import get_gemini
from processor import pipeline, vector_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ResearchRAG API",
    description="Research paper discovery, RAG, teaching chat, and content generation",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    cfg.validate()
    await db.init()
    logger.info("ResearchRAG API ready.")


@app.on_event("shutdown")
async def shutdown():
    await db.close()


# ── Rate limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, min_interval: float):
        self._min = min_interval
        self._last = 0.0

    async def wait(self):
        elapsed = time.time() - self._last
        if elapsed < self._min:
            await asyncio.sleep(self._min - elapsed)
        self._last = time.time()


arxiv_limiter = RateLimiter(cfg.ARXIV_RATE_LIMIT_SECONDS)
pubmed_limiter = RateLimiter(cfg.PUBMED_RATE_LIMIT_SECONDS)

# ── In-memory SSE event queues ─────────────────────────────────────────────────
# Maps paper_id → list of SSE event dicts
_sse_queues: Dict[str, asyncio.Queue] = {}


def _push_event(paper_id: str, event: str, data: Dict):
    q = _sse_queues.get(paper_id)
    if q:
        try:
            q.put_nowait({"event": event, "data": data})
        except asyncio.QueueFull:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# PAPERS — Search, Download, List, Delete
# ══════════════════════════════════════════════════════════════════════════════

class SearchRequest(BaseModel):
    topic: str
    limit: int = 5
    source: str = "both"           # arxiv | pubmed | both
    date_from: Optional[str] = None  # YYYY-MM-DD
    date_to: Optional[str] = None
    sort_by: str = "relevance"     # relevance | date

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
        results = []
        for r in search.results():
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

        # Step 1: search for IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pmc",
            "term": f"{topic} open access[filter]{date_filter}",
            "retmax": limit,
            "retmode": "json",
            "sort": "pub+date",
        }
        resp = requests.get(search_url, params=params, timeout=10)
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        await pubmed_limiter.wait()

        # Step 2: fetch summaries
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        sum_params = {"db": "pmc", "id": ",".join(ids), "retmode": "json"}
        sum_resp = requests.get(summary_url, params=sum_params, timeout=10)
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
    out_path = str(out_dir / f"{paper_id}.pdf")

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
    """Download PubMed BioC JSON. Returns local file path."""
    await pubmed_limiter.wait()
    paper_id = paper_data["paper_id"]
    pmc_id = paper_data.get("pmc_id", paper_id.replace("PMC", ""))
    out_dir = Path(cfg.DOWNLOADS_DIR) / "pubmed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{paper_id}.json")

    if Path(out_path).exists():
        return out_path

    url = f"https://www.ncbi.nlm.nih.gov/research/bioxiv/PMID/{pmc_id}?format=json"
    # BioC endpoint
    bioc_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC{pmc_id}&format=json"

    for attempt in range(1, cfg.MAX_RETRIES + 1):
        try:
            resp = requests.get(
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pmc", "id": pmc_id, "rettype": "json", "retmode": "json"},
                timeout=30,
            )
            # PubMed doesn't always return JSON — fall back to XML→text
            if resp.status_code == 200 and resp.content:
                with open(out_path, "wb") as f:
                    f.write(resp.content)
                return out_path
            raise ValueError(f"Empty response from PubMed (status {resp.status_code})")
        except Exception as exc:
            if attempt < cfg.MAX_RETRIES:
                await asyncio.sleep(cfg.RETRY_BACKOFF_BASE ** (attempt - 1))
            else:
                raise RuntimeError(f"PubMed download failed: {exc}")


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

        # Persist sections + chunks to PostgreSQL
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
            # Update title/abstract from extraction if not already set
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

    _push_event(paper_id, "done", {
        "success": True,
        "paper_id": paper_id,
        "session_id": session_id,
        "chunk_count": len(chunks_data),
    })


@app.post("/api/papers/search")
async def search_papers(req: SearchRequest, background_tasks: BackgroundTasks):
    """
    Search for papers and immediately begin download + processing pipeline.
    Returns paper list; use SSE /api/papers/{paper_id}/progress to track.
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

    # Persist to DB and kick off pipelines
    queued = []
    for paper_data in results:
        pid = paper_data["paper_id"]
        paper_data["topic"] = req.topic

        async with db.session() as session:
            existing = await db.get_paper(session, pid)
            if existing and existing.pipeline_stage == "processed":
                paper_data["pipeline_stage"] = "processed"
                queued.append({**paper_data, "pipeline_stage": "processed", "cached": True})
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

        # Create SSE queue for this paper
        _sse_queues[pid] = asyncio.Queue(maxsize=50)
        background_tasks.add_task(_full_pipeline_task, paper_data, pid)
        queued.append({**paper_data, "pipeline_stage": "pending"})

    return {"papers": queued, "count": len(queued), "topic": req.topic}


@app.get("/api/papers/{paper_id}/progress")
async def paper_progress(paper_id: str):
    """
    Server-Sent Events stream for pipeline progress.
    Frontend: const es = new EventSource('/api/papers/{paper_id}/progress')
    Events: progress, done
    """
    if paper_id not in _sse_queues:
        # Check if paper already processed
        async with db.session() as session:
            paper = await db.get_paper(session, paper_id)
        if paper and paper.pipeline_stage == "processed":
            async def _already_done():
                yield f"event: done\ndata: {json.dumps({'success': True, 'paper_id': paper_id})}\n\n"
            return StreamingResponse(_already_done(), media_type="text/event-stream")
        raise HTTPException(404, "No active pipeline for this paper.")

    q = _sse_queues[paper_id]

    async def event_generator() -> AsyncGenerator[str, None]:
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=30.0)
                data_str = json.dumps(item["data"])
                yield f"event: {item['event']}\ndata: {data_str}\n\n"
                if item["event"] == "done":
                    _sse_queues.pop(paper_id, None)
                    break
            except asyncio.TimeoutError:
                yield "event: heartbeat\ndata: {}\n\n"

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
    async with db.session() as session:
        paper = await db.get_paper(session, paper_id)
        if not paper:
            raise HTTPException(404, "Paper not found.")
        deleted = await db.delete_paper(session, paper_id)
    try:
        vector_store.delete_paper(paper_id)
    except Exception as e:
        logger.warning(f"ChromaDB delete failed for {paper_id}: {e}")
    return {"deleted": deleted, "paper_id": paper_id}


# ══════════════════════════════════════════════════════════════════════════════
# CHAT — Multi-session teaching chat with memory
# ══════════════════════════════════════════════════════════════════════════════

class MessageRequest(BaseModel):
    message: str
    level: Optional[str] = None   # if provided, updates session level

    @validator("message")
    def msg_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty.")
        if len(v) > 2000:
            raise ValueError("Message too long (max 2000 chars).")
        return v


@app.get("/api/chat/sessions")
async def list_sessions():
    """Return all chat sessions for the sidebar."""
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


@app.get("/api/chat/sessions/{session_id}")
async def get_session(session_id: str):
    """Return session info + full message history."""
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
async def send_message(session_id: str, req: MessageRequest):
    """
    Send a message and get a teaching response from Gemini.
    Full conversation history is passed for contextual memory.
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

        # Update level if requested
        effective_level = req.level or sess.level
        if req.level and req.level != sess.level:
            await db.update_session_level(session, session_id, req.level)

    # Get paper context from ChromaDB (most relevant chunks for this message)
    try:
        context_results = vector_store.query(
            query_text=req.message,
            n_results=5,
            filters={"paper_id": paper.paper_id},
        )
        paper_context = "\n\n".join(r["content"] for r in context_results)
        if not paper_context:
            paper_context = vector_store.get_paper_context(paper.paper_id)
    except Exception:
        paper_context = f"{paper.title}\n\n{paper.abstract or ''}"

    # Build history list
    history = [{"role": m.role, "content": m.content} for m in messages]

    # Call Gemini
    try:
        gemini = get_gemini()
        authors_str = ", ".join(paper.authors or []) or "Unknown authors"
        response_text = gemini.chat_response(
            paper_context=paper_context,
            paper_title=paper.title or paper.paper_id,
            paper_authors=authors_str,
            message_history=history,
            user_message=req.message,
            level=effective_level,
        )
    except Exception as exc:
        logger.error(f"Gemini chat failed: {exc}")
        raise HTTPException(503, f"AI response failed: {exc}")

    # Persist both messages
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
    platform: str                   # twitter | linkedin | carousel
    content_style: str = "educational"  # professional | minimal | bold | educational
    carousel_style: Optional[Dict[str, Any]] = None

    @validator("platform")
    def valid_platform(cls, v):
        if v not in ("twitter", "linkedin", "carousel"):
            raise ValueError("platform must be twitter, linkedin, or carousel")
        return v


@app.post("/api/generate")
async def generate_content(req: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Generate social content for a processed paper.
    Returns immediately; SSE /api/generate/{paper_id}/progress streams updates.
    """
    async with db.session() as session:
        paper = await db.get_paper(session, req.paper_id)
        if not paper:
            raise HTTPException(404, "Paper not found.")
        if paper.pipeline_stage != "processed":
            raise HTTPException(400, "Paper must be fully processed before generating content.")
        sections = await db.get_sections(session, req.paper_id)

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
        # Build context
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
            result = gemini.generate_twitter_thread(
                paper.title or "", authors_str,
                paper.abstract or "", key_content,
            )
            content_str = json.dumps(result)
            hashtags = result.get("hashtags", [])

        elif platform == "linkedin":
            result = gemini.generate_linkedin_post(
                paper.title or "", authors_str,
                paper.abstract or "", key_content,
            )
            content_str = json.dumps(result)
            hashtags = result.get("hashtags", [])

        elif platform == "carousel":
            # Generate content via Gemini
            result = gemini.generate_carousel_content(
                paper.title or "", authors_str,
                paper.abstract or "", key_content,
            )
            # Render slides (Playwright)
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

        # Persist to DB
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
        # Signal stream end
        q = _sse_queues.get(queue_key)
        if q:
            q.put_nowait({"event": "done", "data": {}})


@app.get("/api/generate/{queue_key}/progress")
async def generate_progress(queue_key: str):
    """SSE stream for content generation progress."""
    if queue_key not in _sse_queues:
        raise HTTPException(404, "No active generation for this key.")

    q = _sse_queues[queue_key]

    async def stream():
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=60.0)
                yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"
                if item["event"] == "done":
                    _sse_queues.pop(queue_key, None)
                    break
            except asyncio.TimeoutError:
                yield "event: heartbeat\ndata: {}\n\n"

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
    return {"status": "ok", "version": "3.0.0"}