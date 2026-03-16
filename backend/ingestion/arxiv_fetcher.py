"""
arxiv_fetcher.py — Fetches papers from arXiv using the official arxiv Python SDK.

The SDK is synchronous, so all SDK calls are wrapped in run_in_executor
to avoid blocking the event loop.
"""

import asyncio
import logging
from datetime import datetime

import arxiv

from config import cfg
from ingestion.validator import DocumentInput, ValidationError, validate

logger = logging.getLogger(__name__)


def _build_query(topic: str, date_from: datetime | None) -> str:
    """Build the arXiv query string, optionally filtered by submission date."""
    query = topic
    if date_from:
        date_str = date_from.strftime("%Y%m%d")
        query = f"{topic} AND submittedDate:[{date_str}000000 TO 99991231235959]"
    return query


def _map_result(result: arxiv.Result, topic: str) -> dict:
    """Map an arxiv.Result to a raw dict matching DocumentInput fields."""
    return {
        "paper_id":  result.entry_id.split("/")[-1],   # e.g. "2301.07041v1"
        "source":    "arxiv",
        "title":     result.title or "",
        "abstract":  result.summary or "",
        "authors":   [str(a) for a in result.authors],
        "url":       result.entry_id,
        "file_path": "",
        "topic":     topic,
        "extra_metadata": {
            "published":   result.published.isoformat() if result.published else "",
            "updated":     result.updated.isoformat()   if result.updated   else "",
            "categories":  result.categories,
            "pdf_url":     result.pdf_url or "",
            "doi":         result.doi or "",
        },
    }


def _fetch_sync(query: str, limit: int) -> list[arxiv.Result]:
    """Blocking SDK call — runs inside an executor."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    return list(client.results(search))


async def search(
    topic: str,
    limit: int,
    date_from: datetime | None = None,
) -> list[DocumentInput]:
    """
    Search arXiv for papers matching topic.

    Returns a list of validated DocumentInput objects.
    Papers that fail validation are skipped with a warning.
    """
    query = _build_query(topic, date_from)
    logger.info("arXiv search: query=%r limit=%d", query, limit)

    # FIX A1: Rate-limit BEFORE the API call — one sleep per search() invocation.
    # Previously the sleep was inside the per-result loop, causing N * ARXIV_RATE_LIMIT
    # seconds of wasted delay after all results had already been fetched.
    await asyncio.sleep(cfg.ARXIV_RATE_LIMIT)

    # FIX A2: Use get_running_loop() — get_event_loop() is deprecated in Python 3.10+
    # and raises RuntimeError in some Python 3.12 contexts when called from a coroutine.
    loop = asyncio.get_running_loop()
    try:
        results = await loop.run_in_executor(None, _fetch_sync, query, limit)
    except Exception as exc:
        logger.error("arXiv SDK call failed: %s", exc)
        raise

    documents: list[DocumentInput] = []

    for result in results:
        raw = _map_result(result, topic)
        try:
            doc = validate(raw)
        except ValidationError as exc:
            logger.warning("arXiv paper skipped (validation failed): %s — %s", raw.get("paper_id"), exc)
            continue

        documents.append(doc)
        logger.debug("arXiv accepted: %s", doc.paper_id)

    logger.info("arXiv search complete: %d/%d papers passed validation", len(documents), len(results))
    return documents