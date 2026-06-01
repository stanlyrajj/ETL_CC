"""
arxiv_fetcher.py — Fetches papers from arXiv using the official arxiv Python SDK.

The SDK is synchronous so all SDK calls run inside run_in_executor.

Key fixes vs original
---------------------
1. max_results / page size:
   The arXiv SDK always fetches pages of 100 results from the API regardless
   of the max_results argument on arxiv.Search. max_results only controls when
   the iterator *stops*. We use itertools.islice to stop the generator early
   so it never triggers a second HTTP request, which is what causes slow
   responses and 429s on broad queries.

2. 429 handling:
   When arXiv returns HTTP 429, wait ARXIV_429_WAIT seconds then retry once.
   Second 429 → user-friendly "wait and try again" message.

3. Timeout fallback:
   First attempt: ARXIV_FIRST_TIMEOUT seconds, full query.
   On timeout: retry with simplified topic-only query + ARXIV_RETRY_TIMEOUT.

4. In-memory cache:
   Identical queries served from cache for CACHE_TTL seconds (default 10 min).
"""

import asyncio
import hashlib
import itertools
import logging
import time
from datetime import datetime

import arxiv

from config import cfg
from ingestion.validator import DocumentInput, ValidationError, validate

logger = logging.getLogger(__name__)


# ── Cache ─────────────────────────────────────────────────────────────────────

_CACHE_TTL         = 600
_MAX_CACHE_ENTRIES = 50
_cache: dict[str, tuple[list[DocumentInput], float]] = {}


def _cache_key(query: str, limit: int, sort_by: str) -> str:
    return hashlib.sha256(f"{query}|{limit}|{sort_by}".encode()).hexdigest()[:16]


def _cache_get(key: str) -> list[DocumentInput] | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    results, expires_at = entry
    if time.monotonic() > expires_at:
        _cache.pop(key, None)
        return None
    return results


def _cache_set(key: str, results: list[DocumentInput]) -> None:
    if len(_cache) >= _MAX_CACHE_ENTRIES:
        _cache.pop(min(_cache, key=lambda k: _cache[k][1]), None)
    _cache[key] = (results, time.monotonic() + _CACHE_TTL)


# ── Query helpers ─────────────────────────────────────────────────────────────

def _to_arxiv_term(text: str) -> str:
    text = text.strip()
    return f'all:"{text}"' if " " in text else f"all:{text}"


def _build_query(
    topic: str, date_from: datetime | None, date_to: datetime | None,
    category: str | None, keyword: str | None,
) -> str:
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


def _build_fallback_query(topic: str) -> str:
    return _to_arxiv_term(topic)


def _map_result(result: arxiv.Result, topic: str) -> dict:
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


def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "too many requests" in msg or "rate limit" in msg


# ── Sync fetch ────────────────────────────────────────────────────────────────

def _fetch_sync(query: str, max_results: int, sort_by: str) -> list[arxiv.Result]:
    """
    Blocking SDK call — runs in executor.

    IMPORTANT: The arxiv SDK always requests pages of 100 from the API.
    max_results on arxiv.Search is purely an iterator stop condition.
    We use itertools.islice to stop the generator after max_results items
    so the SDK never makes a second HTTP request for a second page.
    This is what actually limits the API call to a single small request.

    num_retries=0: we handle all retries explicitly in the async layer.
    """
    criterion = (
        arxiv.SortCriterion.Relevance
        if sort_by == "relevance"
        else arxiv.SortCriterion.SubmittedDate
    )
    client = arxiv.Client(num_retries=0, delay_seconds=0)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=criterion,
        sort_order=arxiv.SortOrder.Descending,
    )
    # islice stops consuming the generator after max_results items.
    # Without this, list() would exhaust the full iterator causing the SDK
    # to fetch additional 100-result pages even though we only want 15-20.
    return list(itertools.islice(client.results(search), max_results))


# ── Async attempt ─────────────────────────────────────────────────────────────

async def _attempt(query: str, max_results: int, sort_by: str, timeout: float) -> list[arxiv.Result]:
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(None, _fetch_sync, query, max_results, sort_by),
        timeout=timeout,
    )


# ── Validate raw results ──────────────────────────────────────────────────────

def _validate_results(raw_results: list[arxiv.Result], topic: str) -> list[DocumentInput]:
    documents = []
    for result in raw_results:
        raw = _map_result(result, topic)
        try:
            documents.append(validate(raw))
        except ValidationError as exc:
            logger.warning("arXiv paper skipped: %s — %s", raw.get("paper_id"), exc)
    return documents


# ── Public API ────────────────────────────────────────────────────────────────

_ARXIV_429_WAIT = 15   # seconds to wait after a 429 before retrying


async def search(
    topic:     str,
    limit:     int,
    date_from: datetime | None = None,
    date_to:   datetime | None = None,
    sort_by:   str = "date",
    category:  str | None = None,
    keyword:   str | None = None,
) -> list[DocumentInput]:
    first_timeout = getattr(cfg, "ARXIV_FIRST_TIMEOUT", 10)
    retry_timeout = getattr(cfg, "ARXIV_RETRY_TIMEOUT", 8)

    full_query     = _build_query(topic, date_from, date_to, category, keyword)
    fallback_query = _build_fallback_query(topic)

    # Fetch slightly more than needed for validation headroom, cap at 20.
    # Combined with islice in _fetch_sync, this ensures a single HTTP request.
    max_results = min(limit * 3, 20)

    # Cache check
    ck = _cache_key(full_query, max_results, sort_by)
    cached = _cache_get(ck)
    if cached is not None:
        logger.info("arXiv cache hit: %d results for %r", len(cached), full_query)
        return cached[:limit]

    logger.info("arXiv search: query=%r max_results=%d sort=%s", full_query, max_results, sort_by)
    await asyncio.sleep(cfg.ARXIV_RATE_LIMIT)

    raw_results: list[arxiv.Result] = []
    used_fallback = False

    try:
        # ── Attempt 1: full query ──────────────────────────────────────────
        raw_results = await _attempt(full_query, max_results, sort_by, first_timeout)
        logger.info("arXiv attempt 1 succeeded: %d results", len(raw_results))

    except asyncio.TimeoutError:
        logger.warning("arXiv timed out after %ds — retrying with simplified query", first_timeout)
        try:
            # ── Timeout retry: topic only ──────────────────────────────────
            raw_results = await _attempt(fallback_query, max_results, sort_by, retry_timeout)
            used_fallback = True
            logger.info("arXiv fallback succeeded: %d results", len(raw_results))
        except asyncio.TimeoutError:
            raise RuntimeError(
                "arXiv is responding slowly right now. "
                "Try a more specific topic (e.g. 'quantum entanglement' instead of "
                "'quantum mechanics'), remove date filters, or try again in a moment."
            )

    except Exception as exc:
        if _is_rate_limited(exc):
            # ── 429: wait then retry ───────────────────────────────────────
            logger.warning("arXiv returned 429 — waiting %ds before retry", _ARXIV_429_WAIT)
            await asyncio.sleep(_ARXIV_429_WAIT)
            try:
                raw_results = await _attempt(full_query, max_results, sort_by, first_timeout)
                logger.info("arXiv retry after 429 succeeded: %d results", len(raw_results))
            except Exception as retry_exc:
                if _is_rate_limited(retry_exc):
                    raise RuntimeError(
                        "arXiv is rate-limiting this client. "
                        "Please wait 60 seconds and try again, "
                        "or switch the source to PubMed."
                    )
                if isinstance(retry_exc, asyncio.TimeoutError):
                    raise RuntimeError(
                        "arXiv is responding slowly. "
                        "Try a more specific topic or try again in a moment."
                    )
                raise RuntimeError(f"arXiv search failed: {retry_exc}") from retry_exc
        else:
            logger.error("arXiv SDK error: %s", exc)
            raise RuntimeError(f"arXiv search failed: {exc}") from exc

    documents = _validate_results(raw_results, topic)
    logger.info(
        "arXiv complete: %d/%d passed validation%s",
        len(documents), len(raw_results),
        " [simplified fallback]" if used_fallback else "",
    )

    _cache_set(ck, documents)
    return documents[:limit]