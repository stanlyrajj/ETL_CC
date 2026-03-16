"""
pubmed_fetcher.py — Fetches open-access papers from PubMed via NCBI E-utilities.

Uses async httpx for all HTTP calls.
Applies open-access filter to every query.
"""

import asyncio
import logging
from datetime import datetime

import httpx

from config import cfg
from ingestion.validator import DocumentInput, ValidationError, validate

logger = logging.getLogger(__name__)

_ESEARCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_TIMEOUT      = 30.0


def _build_esearch_query(topic: str, date_from: datetime | None) -> str:
    """Build the eSearch query string with open-access filter and optional date range."""
    query = f"({topic}) AND free full text[filter]"
    if date_from:
        date_str = date_from.strftime("%Y/%m/%d")
        query += f' AND ("{date_str}"[Date - Publication] : "3000"[Date - Publication])'
    return query


def _parse_author_list(author_list: list[dict]) -> list[str]:
    """Extract author names from the eSummary AuthorList field."""
    names = []
    for author in author_list:
        name = author.get("Name") or author.get("name") or ""
        if name:
            names.append(name)
    return names


def _map_summary(uid: str, summary: dict, topic: str) -> dict:
    """Map an eSummary record to a raw dict matching DocumentInput fields."""
    authors = _parse_author_list(summary.get("Authors") or summary.get("authors") or [])

    # PubMed article URL
    url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"

    # Best available date field
    pub_date = (
        summary.get("PubDate")
        or summary.get("EPubDate")
        or summary.get("pubdate")
        or ""
    )

    return {
        "paper_id":  f"pubmed-{uid}",
        "source":    "pubmed",
        "title":     summary.get("Title") or summary.get("title") or "",
        "abstract":  "",                   # eSummary does not return abstracts
        "authors":   authors,
        "url":       url,
        "file_path": "",
        "topic":     topic,
        "extra_metadata": {
            "pubmed_id":  uid,
            "pub_date":   pub_date,
            "journal":    summary.get("Source") or "",
            "volume":     summary.get("Volume") or "",
            "issue":      summary.get("Issue") or "",
            "pages":      summary.get("Pages") or "",
            "doi":        summary.get("DOI") or "",
        },
    }


async def _esearch(client: httpx.AsyncClient, query: str, limit: int) -> list[str]:
    """Run eSearch and return a list of PubMed UIDs."""
    params = {
        "db":       "pubmed",
        "term":     query,
        "retmax":   str(limit),
        "retmode":  "json",
        "sort":     "date",
    }
    response = await client.get(_ESEARCH_URL, params=params, timeout=_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    uids: list[str] = data.get("esearchresult", {}).get("idlist", [])
    return uids


async def _esummary(client: httpx.AsyncClient, uids: list[str]) -> dict[str, dict]:
    """Fetch eSummary records for a batch of UIDs. Returns {uid: summary_dict}."""
    params = {
        "db":      "pubmed",
        "id":      ",".join(uids),
        "retmode": "json",
    }
    response = await client.get(_ESUMMARY_URL, params=params, timeout=_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return data.get("result", {})


async def search(
    topic: str,
    limit: int,
    date_from: datetime | None = None,
) -> list[DocumentInput]:
    """
    Search PubMed for open-access papers matching topic.

    Returns a list of validated DocumentInput objects.
    Papers that fail validation are skipped with a warning.
    """
    query = _build_esearch_query(topic, date_from)
    logger.info("PubMed search: query=%r limit=%d", query, limit)

    async with httpx.AsyncClient() as client:
        # Step 1: get UIDs
        await asyncio.sleep(cfg.PUBMED_RATE_LIMIT)
        try:
            uids = await _esearch(client, query, limit)
        except Exception as exc:
            logger.error("PubMed eSearch failed: %s", exc)
            raise

        if not uids:
            logger.info("PubMed search returned no results.")
            return []

        logger.debug("PubMed eSearch returned %d UIDs", len(uids))

        # Step 2: fetch summaries for all UIDs in one batch call
        await asyncio.sleep(cfg.PUBMED_RATE_LIMIT)
        try:
            summaries = await _esummary(client, uids)
        except Exception as exc:
            logger.error("PubMed eSummary failed: %s", exc)
            raise

    # Step 3: validate each result
    documents: list[DocumentInput] = []

    for uid in uids:
        summary = summaries.get(uid)
        if not summary or uid == "uids":          # "uids" is a metadata key NCBI includes
            continue

        raw = _map_summary(uid, summary, topic)
        try:
            doc = validate(raw)
        except ValidationError as exc:
            logger.warning("PubMed paper skipped (validation failed): %s — %s", uid, exc)
            continue

        documents.append(doc)
        logger.debug("PubMed accepted: %s", doc.paper_id)

    logger.info("PubMed search complete: %d/%d papers passed validation", len(documents), len(uids))
    return documents