"""
main.py — FastAPI application entry point.

Currently applies: Critical fixes 1 (auth/binding) and 2 (SQLite atomicity).
Critical 3 (SSE leak) and Critical 4 (upload size limit) will add to this file
when their respective patches are applied.
"""
import asyncio
import logging
import secrets

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
    allow_headers=["*", "X-App-Token"],
)

# ── Token auth middleware ─────────────────────────────────────────────────────
# Runs before every request. Skips /health so monitoring probes still work.
# When APP_TOKEN is empty the middleware is a no-op (loopback-only deployments).
@app.middleware("http")
async def token_auth(request: Request, call_next):
    if not cfg.APP_TOKEN:
        return await call_next(request)

    if request.method == "OPTIONS" or request.url.path == "/health":
        return await call_next(request)

    token = request.headers.get("X-App-Token", "")
    if not secrets.compare_digest(token, cfg.APP_TOKEN):
        logger.warning(
            "Rejected request from %s — invalid or missing X-App-Token",
            request.client.host if request.client else "unknown",
        )
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized. Set X-App-Token header."},
        )

    return await call_next(request)


from api import papers, chat, generate, progress  # noqa: E402
from api import study, technical                   # noqa: E402

app.include_router(papers.router,    prefix="/api")
app.include_router(chat.router,      prefix="/api")
app.include_router(generate.router,  prefix="/api")
app.include_router(progress.router,  prefix="/api")
app.include_router(study.router,     prefix="/api")
app.include_router(technical.router, prefix="/api")


@app.on_event("startup")
async def startup():
    cfg.validate()
    cfg.create_dirs()
    logger.info("Config validated. Directories ready.")

    if cfg.APP_TOKEN:
        logger.info("Token auth enabled (X-App-Token required on all requests).")
    else:
        logger.info(
            "Token auth disabled. Server bound to %s — "
            "set APP_TOKEN in .env if you expose this to a network.",
            cfg.HOST,
        )

    await db.init()
    logger.info("Database initialised.")

    stuck_stages = ("downloading", "downloaded", "processing")
    stuck_papers = []
    async with db.session() as sess:
        for stage in stuck_stages:
            found = await db.list_papers(sess, stage=stage, limit=200)
            stuck_papers.extend(found)

    if stuck_papers:
        async with db.session() as sess:
            for paper in stuck_papers:
                await db.set_stage(sess, paper.paper_id, "pending")
        logger.info("Reset %d stuck paper(s) to 'pending'.", len(stuck_papers))
        from api.papers import _run_pipeline
        from api.progress import get_or_create_paper_queue
        from ingestion.validator import DocumentInput
        for paper in stuck_papers:
            doc = DocumentInput(
                paper_id=paper.paper_id, source=paper.source or "local",
                title=paper.title or "", abstract=paper.abstract or "",
                authors=paper.authors or [], url=paper.url or "",
                file_path=paper.file_path or "", topic=paper.topic or "",
            )
            get_or_create_paper_queue(paper.paper_id)
            asyncio.create_task(_run_pipeline(doc))

    logger.info("ResearchRAG startup complete.")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=cfg.HOST, port=cfg.PORT, reload=False)