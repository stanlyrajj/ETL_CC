"""
generate.py — Social content generation, PDF export, share deeplink, and model
selection endpoints.

Every route returns explicit success or failure — no silent failures.
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from config import cfg, AVAILABLE_MODELS
from database import SocialContent, db
from api.progress import get_or_create_generate_queue, push_generate_event
from content.carousel      import generate as carousel_generate
from content.linkedin_post import generate as linkedin_generate
from content.twitter       import generate as twitter_generate
from export.pdf_renderer   import render as render_pdf
from export.share          import linkedin_deeplink, twitter_deeplink

logger = logging.getLogger(__name__)

router = APIRouter()

_VALID_PLATFORMS = ("twitter", "linkedin", "carousel")


# ── Pydantic models ───────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    paper_id:     str = Field(..., min_length=1)
    platform:     str = Field(...)
    style:        str = Field("educational")
    tone:         str = Field("conversational")
    color_scheme: str = Field("light")


class ModelSelectRequest(BaseModel):
    model_id: str = Field(..., min_length=1)


# ── Model selection routes ────────────────────────────────────────────────────

@router.get("/models")
async def list_models():
    """
    Return the curated list of available models.

    When LLM_PROVIDER=openrouter, returns the full curated free model list
    with the currently active model marked. For other providers, returns
    a single entry showing the configured model.
    """
    provider = cfg.LLM_PROVIDER.lower()

    if provider == "openrouter":
        models = [
            {**m, "active": m["id"] == cfg.LLM_MODEL}
            for m in AVAILABLE_MODELS
        ]
        return {
            "provider":      provider,
            "active_model":  cfg.LLM_MODEL,
            "models":        models,
        }

    # For non-OpenRouter providers, show the single configured model
    return {
        "provider":     provider,
        "active_model": cfg.LLM_MODEL,
        "models": [
            {
                "id":          cfg.LLM_MODEL,
                "name":        cfg.LLM_MODEL,
                "provider":    provider,
                "description": f"Configured via LLM_MODEL in .env",
                "recommended": True,
                "active":      True,
            }
        ],
    }


@router.post("/models/select")
async def select_model(request: ModelSelectRequest):
    """
    Switch the active LLM model at runtime.

    Only permitted when LLM_PROVIDER=openrouter — switching models between
    providers (e.g. from Gemini to OpenAI) requires a server restart since
    it changes the underlying SDK and authentication.

    The selected model must be in the curated AVAILABLE_MODELS list.
    """
    if cfg.LLM_PROVIDER.lower() != "openrouter":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model switching is only available when LLM_PROVIDER=openrouter. "
                f"Current provider: {cfg.LLM_PROVIDER!r}. "
                "To change the model for other providers, update LLM_MODEL in .env and restart."
            ),
        )

    valid_ids = {m["id"] for m in AVAILABLE_MODELS}
    if request.model_id not in valid_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model {request.model_id!r} is not in the curated list. "
                f"Available models: {sorted(valid_ids)}"
            ),
        )

    from llm.factory import set_model
    set_model(request.model_id)

    model_info = next(m for m in AVAILABLE_MODELS if m["id"] == request.model_id)
    logger.info("Model switched to: %s", request.model_id)

    return {
        "success":      True,
        "active_model": request.model_id,
        "model":        model_info,
        "message":      f"Switched to {model_info['name']}.",
    }


# ── Background generation task ────────────────────────────────────────────────

async def _run_generate(request: GenerateRequest, queue_key: str) -> None:
    """
    Run content generation in the background.
    Pushes SSE events: started, completed, failed, done.
    Never raises — all failures are caught and pushed as events.
    """
    platform = request.platform.lower()

    await push_generate_event(queue_key, "started", {
        "queue_key": queue_key,
        "paper_id":  request.paper_id,
        "platform":  platform,
    })

    try:
        if platform == "twitter":
            result = await twitter_generate(request.paper_id, request.style, request.tone)

        elif platform == "linkedin":
            result = await linkedin_generate(request.paper_id, request.style, request.tone)

        elif platform == "carousel":
            result = await carousel_generate(
                request.paper_id, request.style, request.tone, request.color_scheme
            )

        else:
            raise ValueError(f"Unknown platform: {platform!r}")

        await push_generate_event(queue_key, "completed", {
            "queue_key": queue_key,
            "paper_id":  request.paper_id,
            "platform":  platform,
            "result":    result,
        })
        await push_generate_event(queue_key, "done", {
            "queue_key": queue_key,
            "success":   True,
        })
        logger.info("Generation complete: paper=%s platform=%s", request.paper_id, platform)

    except Exception as exc:
        error = str(exc)
        logger.error("Generation failed: paper=%s platform=%s: %s",
                     request.paper_id, platform, error)
        await push_generate_event(queue_key, "failed", {
            "queue_key": queue_key,
            "paper_id":  request.paper_id,
            "platform":  platform,
            "error":     error,
        })
        await push_generate_event(queue_key, "done", {
            "queue_key": queue_key,
            "success":   False,
            "error":     error,
        })


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/generate")
async def generate_content(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Queue content generation as a background task.
    Returns a queue_key for SSE progress tracking.
    """
    platform = request.platform.lower()
    if platform not in _VALID_PLATFORMS:
        raise HTTPException(
            status_code=400,
            detail=f"platform must be one of {_VALID_PLATFORMS}. Got: {platform!r}",
        )

    async with db.session() as sess:
        paper = await db.get_paper(sess, request.paper_id)
    if paper is None:
        raise HTTPException(
            status_code=404,
            detail=f"Paper not found: {request.paper_id!r}",
        )
    if paper.pipeline_stage != "processed":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Paper is not ready yet. "
                f"Current stage: {paper.pipeline_stage!r}. "
                "Wait for pipeline_stage to be 'processed'."
            ),
        )

    queue_key = f"{request.paper_id}_{platform}_{uuid.uuid4().hex[:8]}"
    get_or_create_generate_queue(queue_key)
    background_tasks.add_task(_run_generate, request, queue_key)

    return {
        "queue_key": queue_key,
        "paper_id":  request.paper_id,
        "platform":  platform,
        "message":   "Generation started. Track progress via SSE.",
    }


@router.get("/generate/history/{paper_id}")
async def generation_history(paper_id: str, platform: Optional[str] = None):
    """Return past generation outputs for a paper."""
    async with db.session() as sess:
        paper = await db.get_paper(sess, paper_id)
        if paper is None:
            raise HTTPException(
                status_code=404,
                detail=f"Paper not found: {paper_id!r}",
            )
        items = await db.list_social(sess, paper_id, platform=platform)

    return {
        "paper_id": paper_id,
        "items": [
            {
                "id":           item.id,
                "platform":     item.platform,
                "content_type": item.content_type,
                "content":      item.content,
                "hashtags":     item.hashtags or [],
                "created_at":   item.created_at.isoformat() if item.created_at else None,
            }
            for item in items
        ],
    }


@router.post("/generate/{content_id}/export")
async def export_carousel(content_id: int):
    """Render a carousel SocialContent record to a PDF file."""
    async with db.session() as sess:
        result = await sess.execute(
            select(SocialContent).where(SocialContent.id == content_id)
        )
        item = result.scalar_one_or_none()

    if item is None:
        raise HTTPException(status_code=404, detail=f"Content not found: id={content_id}")
    if item.platform != "carousel":
        raise HTTPException(
            status_code=400,
            detail=f"Export is only supported for carousel content. Got platform: {item.platform!r}",
        )

    try:
        slides = json.loads(item.content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not parse carousel content: {exc}")

    if not slides:
        raise HTTPException(status_code=400, detail="Carousel has no slides to render.")

    try:
        pdf_path = render_pdf(slides, color_scheme="light", output_dir=cfg.CAROUSEL_OUTPUT_DIR)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF rendering failed: {exc}")

    filename = Path(pdf_path).name
    return {
        "success":      True,
        "content_id":   content_id,
        "file_path":    pdf_path,
        "filename":     filename,
        "download_url": f"/api/generate/{content_id}/download?filename={filename}",
    }


@router.get("/generate/{content_id}/download")
async def download_carousel(content_id: int, filename: Optional[str] = None):
    """Return a carousel PDF as a file download."""
    output_dir = Path(cfg.CAROUSEL_OUTPUT_DIR)
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="No carousel outputs found.")

    if filename:
        safe_name = Path(filename).name
        pdf_path  = output_dir / safe_name
        if not pdf_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {safe_name}. Run the export endpoint first.",
            )
    else:
        pdfs = sorted(output_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not pdfs:
            raise HTTPException(
                status_code=404,
                detail="No PDF files found. Run POST /generate/{content_id}/export first.",
            )
        pdf_path = pdfs[0]

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
    )


@router.get("/generate/{content_id}/share")
async def share_content(content_id: int):
    """Return LinkedIn and Twitter deep-link URLs for a content record."""
    async with db.session() as sess:
        result = await sess.execute(
            select(SocialContent).where(SocialContent.id == content_id)
        )
        item = result.scalar_one_or_none()

    if item is None:
        raise HTTPException(status_code=404, detail=f"Content not found: id={content_id}")

    hashtags: list[str] = item.hashtags or []

    if item.platform == "twitter":
        try:
            tweets = json.loads(item.content)
            first_tweet = tweets[0] if tweets else ""
        except (json.JSONDecodeError, TypeError, IndexError):
            first_tweet = item.content or ""
        tw_url = twitter_deeplink(first_tweet, hashtags)
        li_url = linkedin_deeplink(first_tweet, hashtags)

    elif item.platform == "linkedin":
        li_url = linkedin_deeplink(item.content or "", hashtags)
        tw_url = twitter_deeplink((item.content or "")[:240], hashtags)

    elif item.platform == "carousel":
        try:
            slides = json.loads(item.content)
            summary = " | ".join(
                s.get("title", "") for s in slides[:3] if s.get("title")
            )
        except (json.JSONDecodeError, TypeError):
            summary = "Check out this research carousel"
        li_url = linkedin_deeplink(summary, hashtags)
        tw_url = twitter_deeplink(summary[:200], hashtags)

    else:
        li_url = linkedin_deeplink(item.content or "", hashtags)
        tw_url = twitter_deeplink((item.content or "")[:240], hashtags)

    return {
        "content_id":   content_id,
        "platform":     item.platform,
        "linkedin_url": li_url,
        "twitter_url":  tw_url,
    }