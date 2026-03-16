"""
main.py — FastAPI application entry point.

Registers all routers, configures CORS, runs startup tasks.
"""

import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import cfg
from database import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="ResearchRAG", version="1.0.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cfg.CORS_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers (imported at module level so FastAPI discovers them on load) ──────
from api import papers, chat, generate, progress   # noqa: E402

app.include_router(papers.router,   prefix="/api")
app.include_router(chat.router,     prefix="/api")
app.include_router(generate.router, prefix="/api")
app.include_router(progress.router, prefix="/api")


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    # 1. Validate config and create output directories
    cfg.validate()
    cfg.create_dirs()
    logger.info("Config validated. Directories ready.")

    # 2. Initialise database (creates tables if they don't exist)
    await db.init()
    logger.info("Database initialised.")

    # 3. Reset any papers stuck in intermediate stages back to pending
    #    and re-queue them so they complete on restart.
    stuck_stages = ("downloading", "downloaded", "processing")
    requeued = 0

    async with db.session() as sess:
        for stage in stuck_stages:
            stuck = await db.list_papers(sess, stage=stage, limit=200)
            for paper in stuck:
                await db.set_stage(sess, paper.paper_id, "pending")
                requeued += 1

    if requeued:
        logger.info("Reset %d stuck paper(s) to 'pending' for re-processing.", requeued)

        from api.papers import _run_pipeline
        from api.progress import get_or_create_paper_queue
        from ingestion.validator import DocumentInput

        async with db.session() as sess:
            pending = await db.list_papers(sess, stage="pending", limit=200)

        for paper in pending:
            doc = DocumentInput(
                paper_id=paper.paper_id,
                source=paper.source or "local",
                title=paper.title or "",
                abstract=paper.abstract or "",
                authors=paper.authors or [],
                url=paper.url or "",
                file_path=paper.file_path or "",
                topic=paper.topic or "",
            )
            get_or_create_paper_queue(paper.paper_id)
            asyncio.create_task(_run_pipeline(doc))

    logger.info("ResearchRAG startup complete.")


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Simple liveness check."""
    return {"status": "ok", "version": "1.0.0"}