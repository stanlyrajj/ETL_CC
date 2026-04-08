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


def _to_arxiv_term(text: str) -> str:
    """
    Convert a user-supplied topic or keyword to a valid arXiv query term.

    arXiv query syntax rules:
      - Single words can be used bare:            machine
      - Multi-word phrases must be double-quoted: "machine learning"
      - Prefix with all: to search all fields (title, abstract, author, ...)

    We always use all: so the search matches titles AND abstracts, not just
    the title field (the SDK default when no prefix is given).
    """
    text = text.strip()
    # Quote phrases that contain spaces so arXiv treats them as a single term
    if " " in text:
        return f'all:"{text}"'
    return f"all:{text}"


def _build_query(
    topic:     str,
    date_from: datetime | None,
    date_to:   datetime | None,
    category:  str | None,
    keyword:   str | None,
) -> str:
    """Build the arXiv query string with optional filters."""
    parts = [_to_arxiv_term(topic)]

    if keyword:
        parts.append(_to_arxiv_term(keyword))

    query = " AND ".join(parts)

    if category:
        query = f"cat:{category} AND ({query})"

    if date_from or date_to:
        from_str = date_from.strftime("%Y%m%d") if date_from else "00000101"
        to_str   = date_to.strftime("%Y%m%d")   if date_to   else "99991231"
        query += f" AND submittedDate:[{from_str}000000 TO {to_str}235959]"

    return query


def _map_result(result: arxiv.Result, topic: str) -> dict:
    """Map an arxiv.Result to a raw dict matching DocumentInput fields."""
    return {
        "paper_id":  result.entry_id.split("/")[-1],
        "source":    "arxiv",
        "title":     result.title or "",
        "abstract":  result.summary or "",
        "authors":   [str(a) for a in result.authors],
        "url":       result.entry_id,
        "file_path": "",
        "topic":     topic,
        "extra_metadata": {
            "published":  result.published.isoformat() if result.published else "",
            "updated":    result.updated.isoformat()   if result.updated   else "",
            "categories": result.categories,
            "pdf_url":    result.pdf_url or "",
            "doi":        result.doi or "",
        },
    }


_ARXIV_TIMEOUT = 30  # seconds before giving up on the arXiv API


def _fetch_sync(query: str, limit: int, sort_by: str) -> list[arxiv.Result]:
    """Blocking SDK call — runs inside an executor."""
    criterion = (
        arxiv.SortCriterion.Relevance
        if sort_by == "relevance"
        else arxiv.SortCriterion.SubmittedDate
    )
    # delay_seconds=0 disables the SDK's own inter-page delay (we handle rate
    # limiting ourselves). num_retries=1 stops silent infinite retry loops.
    client = arxiv.Client(
        num_retries=1,
        delay_seconds=0,
    )
    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=criterion,
        sort_order=arxiv.SortOrder.Descending,
    )
    return list(client.results(search))


async def search(
    topic:     str,
    limit:     int,
    date_from: datetime | None = None,
    date_to:   datetime | None = None,
    sort_by:   str = "date",        # "date" | "relevance"
    category:  str | None = None,   # arXiv category code e.g. "cs.LG"
    keyword:   str | None = None,   # must-include term
) -> list[DocumentInput]:
    """
    Search arXiv for papers matching topic and optional filters.

    Returns a list of validated DocumentInput objects.
    Papers that fail validation are skipped with a warning.
    """
    query = _build_query(topic, date_from, date_to, category, keyword)
    logger.info("arXiv search: query=%r limit=%d sort=%s", query, limit, sort_by)

    await asyncio.sleep(cfg.ARXIV_RATE_LIMIT)

    loop = asyncio.get_running_loop()
    try:
        results = await asyncio.wait_for(
            loop.run_in_executor(None, _fetch_sync, query, limit, sort_by),
            timeout=_ARXIV_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error(
            "arXiv search timed out after %ds — the arXiv API may be slow or unreachable. "
            "Query: %r",
            _ARXIV_TIMEOUT, query,
        )
        raise RuntimeError(
            f"arXiv API did not respond within {_ARXIV_TIMEOUT} seconds. "
            "Check your internet connection or try again later."
        )
    except Exception as exc:
        logger.error("arXiv SDK call failed: %s", exc)
        raise

    documents: list[DocumentInput] = []

    for result in results:
        raw = _map_result(result, topic)
        try:
            doc = validate(raw)
        except ValidationError as exc:
            logger.warning(
                "arXiv paper skipped (validation failed): %s — %s",
                raw.get("paper_id"), exc,
            )
            continue
        documents.append(doc)
        logger.debug("arXiv accepted: %s", doc.paper_id)

    logger.info(
        "arXiv search complete: %d/%d papers passed validation",
        len(documents), len(results),
    )
    return documents