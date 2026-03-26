"""
pubmed_fetcher.py — Fetches open-access papers from PubMed via NCBI E-utilities.

Uses async httpx for all HTTP calls.
Applies open-access filter to every query.

Full-text strategy:
  1. eFetch abstract — populates the abstract field (eSummary does not return abstracts).
  2. fetch_bioc_json() — attempts to download full-text BioC JSON from NCBI's
     BioC API. Sets extra_metadata["bioc_path"] on success so the pipeline
     can extract and embed the full paper body, not just the abstract.
     Falls back gracefully if NCBI has no full text for the article.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import httpx

from config import cfg
from ingestion.validator import DocumentInput, ValidationError, validate

logger = logging.getLogger(__name__)

_ESEARCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_EFETCH_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_BIOC_URL     = "https://www.ncbi.nlm.nih.gov/research/bel/api/v1/documents/pubmed/{pmid}"
_TIMEOUT      = 30.0


def _build_esearch_query(
    topic:     str,
    date_from: datetime | None,
    date_to:   datetime | None,
    keyword:   str | None,
) -> str:
    """Build the eSearch query string with open-access filter and optional filters."""
    parts = [f"({topic})"]

    if keyword:
        parts.append(f'("{keyword}"[Title/Abstract])')

    parts.append("free full text[filter]")
    query = " AND ".join(parts)

    if date_from or date_to:
        from_str = date_from.strftime("%Y/%m/%d") if date_from else "1900/01/01"
        to_str   = date_to.strftime("%Y/%m/%d")   if date_to   else "3000/12/31"
        query += (
            f' AND ("{from_str}"[Date - Publication] : "{to_str}"[Date - Publication])'
        )

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
    authors  = _parse_author_list(summary.get("Authors") or summary.get("authors") or [])
    url      = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
    pub_date = (
        summary.get("PubDate") or summary.get("EPubDate")
        or summary.get("pubdate") or ""
    )
    return {
        "paper_id":  f"pubmed-{uid}",
        "source":    "pubmed",
        "title":     summary.get("Title") or summary.get("title") or "",
        "abstract":  "",   # populated by _fetch_abstracts()
        "authors":   authors,
        "url":       url,
        "file_path": "",
        "topic":     topic,
        "extra_metadata": {
            "pubmed_id": uid,
            "pub_date":  pub_date,
            "journal":   summary.get("Source") or "",
            "volume":    summary.get("Volume") or "",
            "issue":     summary.get("Issue") or "",
            "pages":     summary.get("Pages") or "",
            "doi":       summary.get("DOI") or "",
        },
    }


async def _esearch(
    client:  httpx.AsyncClient,
    query:   str,
    limit:   int,
    sort_by: str,
) -> list[str]:
    """Run eSearch and return a list of PubMed UIDs."""
    params = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  str(limit),
        "retmode": "json",
        "sort":    "relevance" if sort_by == "relevance" else "date",
    }
    response = await client.get(_ESEARCH_URL, params=params, timeout=_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return data.get("esearchresult", {}).get("idlist", [])


async def _esummary(
    client: httpx.AsyncClient,
    uids:   list[str],
) -> dict[str, dict]:
    """Fetch eSummary records for a batch of UIDs."""
    params = {
        "db":      "pubmed",
        "id":      ",".join(uids),
        "retmode": "json",
    }
    response = await client.get(_ESUMMARY_URL, params=params, timeout=_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return data.get("result", {})


async def _fetch_abstracts(
    client: httpx.AsyncClient,
    uids:   list[str],
) -> dict[str, str]:
    """
    Fetch abstracts for a batch of UIDs via eFetch (plain text mode).

    eSummary does not return abstracts. This fills that gap.
    Returns a dict mapping uid → abstract string.
    """
    params = {
        "db":       "pubmed",
        "id":       ",".join(uids),
        "rettype":  "abstract",
        "retmode":  "text",
    }
    try:
        await asyncio.sleep(cfg.PUBMED_RATE_LIMIT)
        response = await client.get(_EFETCH_URL, params=params, timeout=_TIMEOUT)
        response.raise_for_status()
        raw_text = response.text
    except Exception as exc:
        logger.warning("eFetch abstracts failed for %d UIDs: %s", len(uids), exc)
        return {}

    # eFetch plain-text format separates records with blank lines between them.
    # Each record starts with "PMID- {uid}" somewhere in the block, and the
    # abstract follows "AB  - ". We parse this simple format directly.
    abstracts: dict[str, str] = {}
    current_uid: str | None   = None
    current_ab_lines: list[str] = []
    in_abstract = False

    for line in raw_text.splitlines():
        if line.startswith("PMID- "):
            # Save previous record
            if current_uid and current_ab_lines:
                abstracts[current_uid] = " ".join(current_ab_lines).strip()
            current_uid     = line[6:].strip()
            current_ab_lines = []
            in_abstract      = False

        elif line.startswith("AB  - "):
            in_abstract = True
            current_ab_lines.append(line[6:].strip())

        elif in_abstract and line.startswith("      "):
            # Continuation lines for the abstract are indented with 6 spaces
            current_ab_lines.append(line.strip())

        elif in_abstract and line.strip() == "":
            # Blank line ends the abstract block
            in_abstract = False

    # Save last record
    if current_uid and current_ab_lines:
        abstracts[current_uid] = " ".join(current_ab_lines).strip()

    logger.debug("eFetch returned abstracts for %d/%d UIDs", len(abstracts), len(uids))
    return abstracts


async def fetch_bioc_json(
    pmid:       str,
    paper_id:   str,
    output_dir: str,
) -> str | None:
    """
    Attempt to download full-text BioC JSON for a PubMed article from NCBI.

    NCBI's BioC API provides full structured article text for open-access
    PubMed Central articles. Not all PubMed articles have full text available —
    this returns None gracefully when full text is not available.

    Parameters
    ----------
    pmid       : PubMed ID string (digits only)
    paper_id   : our internal paper_id, used for the output filename
    output_dir : directory to save the JSON file (downloads/pubmed/)

    Returns
    -------
    Path to the saved JSON file on success, None if full text is unavailable.
    """
    url = _BIOC_URL.format(pmid=pmid)

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=_TIMEOUT) as client:
            await asyncio.sleep(cfg.PUBMED_RATE_LIMIT)
            response = await client.get(url)

            # 404 means no full text available for this article — not an error
            if response.status_code == 404:
                logger.info(
                    "No BioC full text available for PMID=%s (404) — abstract only", pmid
                )
                return None

            response.raise_for_status()

            # Validate it is actually JSON before saving
            data = response.json()
            if not data or not isinstance(data, dict):
                logger.warning("BioC response for PMID=%s is not a valid JSON object", pmid)
                return None

            # Must have at least one document with at least one passage
            documents = data.get("documents", [])
            if not documents:
                logger.info("BioC response for PMID=%s has no documents — abstract only", pmid)
                return None

            total_passages = sum(len(d.get("passages", [])) for d in documents)
            if total_passages == 0:
                logger.info(
                    "BioC response for PMID=%s has no passages — abstract only", pmid
                )
                return None

            # Save to disk
            dest_dir  = Path(output_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{paper_id}.json"
            dest_path.write_text(response.text, encoding="utf-8")

            logger.info(
                "BioC full text saved: PMID=%s → %s (%d passages)",
                pmid, dest_path, total_passages,
            )
            return str(dest_path)

    except httpx.HTTPStatusError as exc:
        logger.warning(
            "BioC fetch HTTP error for PMID=%s: %s — falling back to abstract only",
            pmid, exc,
        )
        return None
    except Exception as exc:
        logger.warning(
            "BioC fetch failed for PMID=%s: %s — falling back to abstract only",
            pmid, exc,
        )
        return None


async def search(
    topic:     str,
    limit:     int,
    date_from: datetime | None = None,
    date_to:   datetime | None = None,
    sort_by:   str = "date",
    keyword:   str | None = None,
) -> list[DocumentInput]:
    """
    Search PubMed for open-access papers matching topic and optional filters.

    Returns a list of validated DocumentInput objects.
    Abstracts are populated via eFetch (eSummary does not return them).
    Papers that fail validation are skipped with a warning.
    """
    query = _build_esearch_query(topic, date_from, date_to, keyword)
    logger.info("PubMed search: query=%r limit=%d sort=%s", query, limit, sort_by)

    async with httpx.AsyncClient() as client:
        await asyncio.sleep(cfg.PUBMED_RATE_LIMIT)
        try:
            uids = await _esearch(client, query, limit, sort_by)
        except Exception as exc:
            logger.error("PubMed eSearch failed: %s", exc)
            raise

        if not uids:
            logger.info("PubMed search returned no results.")
            return []

        logger.debug("PubMed eSearch returned %d UIDs", len(uids))

        await asyncio.sleep(cfg.PUBMED_RATE_LIMIT)
        try:
            summaries = await _esummary(client, uids)
        except Exception as exc:
            logger.error("PubMed eSummary failed: %s", exc)
            raise

        # Fetch abstracts via eFetch — eSummary does not return abstract text
        abstracts = await _fetch_abstracts(client, uids)

    documents: list[DocumentInput] = []

    for uid in uids:
        summary = summaries.get(uid)
        if not summary or uid == "uids":
            continue

        raw = _map_summary(uid, summary, topic)

        # Populate abstract from eFetch result
        raw["abstract"] = abstracts.get(uid, "")

        try:
            doc = validate(raw)
        except ValidationError as exc:
            logger.warning(
                "PubMed paper skipped (validation failed): %s — %s", uid, exc
            )
            continue

        documents.append(doc)
        logger.debug("PubMed accepted: %s", doc.paper_id)

    logger.info(
        "PubMed search complete: %d/%d papers passed validation",
        len(documents), len(uids),
    )
    return documents