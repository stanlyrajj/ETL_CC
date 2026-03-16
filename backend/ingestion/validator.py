"""
validator.py — Validates and sanitizes all incoming document data.

This is the security boundary for the ingestion layer.
All fetchers must pass data through validate() before returning it.
Everything downstream trusts that this has already run.
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Output type ───────────────────────────────────────────────────────────────

@dataclass
class DocumentInput:
    paper_id:       str
    source:         str
    title:          str
    abstract:       str
    authors:        list[str]
    url:            str
    file_path:      str
    topic:          str
    extra_metadata: dict = field(default_factory=dict)


# ── Exception ─────────────────────────────────────────────────────────────────

class ValidationError(Exception):
    pass


# ── Internal sanitization helpers ─────────────────────────────────────────────

# FIX V2: Removed the {0,200} quantifier that previously allowed HTML tags with
# more than 200 characters of attribute content to bypass stripping.
# The correct pattern matches any tag regardless of attribute length.
_HTML_TAG = re.compile(r"<[^>]+>")

# Matches ASCII control characters (except tab/newline/carriage-return)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Prompt-injection phrases to remove (case-insensitive)
_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+previous\s+instructions?|"
    r"disregard\s+(all\s+)?previous|"
    r"forget\s+(all\s+)?previous|"
    r"system\s*:|"
    r"you\s+are\s+now|"
    r"act\s+as\s+(if\s+you\s+(are|were)|an?\s)|"
    r"pretend\s+(you\s+are|to\s+be)|"
    r"new\s+instructions?\s*:|"
    r"override\s+(previous\s+)?instructions?)",
    flags=re.IGNORECASE,
)

# paper_id: keep alphanumeric, hyphen, dot, underscore only
_SAFE_ID_CHARS = re.compile(r"[^a-zA-Z0-9\-._]")


def _sanitize_text(text: str) -> str:
    """Strip HTML tags, control characters, null bytes, and injection patterns."""
    text = text.replace("\x00", "")           # null bytes first
    text = _HTML_TAG.sub("", text)            # remove HTML tags
    text = _CONTROL_CHARS.sub("", text)       # remove control chars
    text = _INJECTION_PATTERNS.sub("", text)  # remove injection phrases
    return text.strip()


def _sanitize_path(path: str) -> str:
    """
    Sanitize a filesystem path — strip only null bytes and control characters.

    FIX V3: file_path must NOT go through _sanitize_text because the injection
    pattern stripper would corrupt legitimate path components.  For example, a
    directory named 'system:config' would be silently mangled to 'config'.
    File paths are trusted values generated internally (local_uploader) or set
    to "" (remote fetchers) — they do not need injection stripping.
    """
    path = path.replace("\x00", "")
    path = _CONTROL_CHARS.sub("", path)
    return path.strip()


def _sanitize_id(paper_id: str) -> str:
    """Keep only safe characters in paper_id and truncate to 100 chars."""
    safe = _SAFE_ID_CHARS.sub("", paper_id)
    return safe[:100]


# ── Public API ────────────────────────────────────────────────────────────────

def validate(data: dict) -> DocumentInput:
    """
    Validate and sanitize raw document data.

    Raises ValidationError with a clear message on any problem.
    Returns a clean DocumentInput on success.
    """
    # ── Required fields ───────────────────────────────────────────────────────
    for field_name in ("paper_id", "source", "title"):
        value = data.get(field_name, "")
        if not value or not str(value).strip():
            raise ValidationError(f"Required field '{field_name}' is missing or empty.")

    # ── Sanitize paper_id ─────────────────────────────────────────────────────
    paper_id = _sanitize_id(str(data["paper_id"]).strip())
    if not paper_id:
        raise ValidationError("paper_id contains no valid characters after sanitization.")

    # ── Sanitize and size-check text fields ───────────────────────────────────
    title = _sanitize_text(str(data["title"]))
    if not title:
        raise ValidationError("title is empty after sanitization.")
    if len(title) > 500:
        raise ValidationError(f"title exceeds 500 characters (got {len(title)}).")

    abstract = _sanitize_text(str(data.get("abstract") or ""))
    if len(abstract) > 5000:
        raise ValidationError(f"abstract exceeds 5000 characters (got {len(abstract)}).")

    # ── Authors: sanitize each entry, filter blanks ───────────────────────────
    raw_authors = data.get("authors") or []
    if not isinstance(raw_authors, list):
        raw_authors = [str(raw_authors)]
    authors = [_sanitize_text(str(a)) for a in raw_authors]
    authors = [a for a in authors if a]

    # ── Plain string fields ───────────────────────────────────────────────────
    source    = _sanitize_text(str(data.get("source") or ""))
    url       = _sanitize_text(str(data.get("url") or ""))
    topic     = _sanitize_text(str(data.get("topic") or ""))

    # FIX V3: Use path-specific sanitizer — injection stripping must not run on
    # file_path because it would corrupt legitimate path components.
    file_path = _sanitize_path(str(data.get("file_path") or ""))

    # ── Extra metadata: keep as-is (callers control this dict) ───────────────
    extra_metadata = data.get("extra_metadata") or {}
    if not isinstance(extra_metadata, dict):
        extra_metadata = {}

    return DocumentInput(
        paper_id=paper_id,
        source=source,
        title=title,
        abstract=abstract,
        authors=authors,
        url=url,
        file_path=file_path,
        topic=topic,
        extra_metadata=extra_metadata,
    )