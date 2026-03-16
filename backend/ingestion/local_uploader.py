"""
local_uploader.py — Handles local PDF file uploads.

Validates file type and size, saves to disk, and returns a DocumentInput.
"""

import logging
import re
import uuid
from pathlib import Path

from fastapi import UploadFile

from config import cfg
from ingestion.validator import DocumentInput, ValidationError, validate

logger = logging.getLogger(__name__)

_MAX_BYTES       = 50 * 1024 * 1024      # 50 MB
_ALLOWED_SUFFIX  = ".pdf"
_SAFE_FILENAME   = re.compile(r"[^a-zA-Z0-9\-_]")  # chars to strip from filename


def _make_paper_id(filename: str) -> str:
    """
    Build a safe, unique paper_id from the uploaded filename.

    Format: local-<sanitized_stem>-<short_uuid>
    """
    stem = Path(filename).stem
    safe_stem = _SAFE_FILENAME.sub("", stem)[:60]   # strip unsafe chars, cap length
    short_id  = uuid.uuid4().hex[:8]
    return f"local-{safe_stem}-{short_id}" if safe_stem else f"local-{short_id}"


async def handle_upload(file: UploadFile, topic: str) -> DocumentInput:
    """
    Accept a PDF upload, validate it, save it to disk, and return a DocumentInput.

    Raises ValidationError if the file type, size, or content is unacceptable.
    """
    # ── File type check ───────────────────────────────────────────────────────
    filename = file.filename or ""
    suffix   = Path(filename).suffix.lower()

    if suffix != _ALLOWED_SUFFIX:
        raise ValidationError(
            f"Unsupported file type '{suffix}'. Only PDF files are accepted."
        )

    # ── Read file content (enforce size limit) ────────────────────────────────
    content = await file.read()

    if len(content) > _MAX_BYTES:
        size_mb = len(content) / (1024 * 1024)
        raise ValidationError(
            f"File is too large ({size_mb:.1f} MB). Maximum allowed size is 50 MB."
        )

    if len(content) == 0:
        raise ValidationError("Uploaded file is empty.")

    # ── Confirm PDF magic bytes ───────────────────────────────────────────────
    if not content.startswith(b"%PDF"):
        raise ValidationError("File does not appear to be a valid PDF (missing PDF header).")

    # ── Build and validate DocumentInput BEFORE writing to disk ──────────────
    # FIX L1: validate() is called here, before any file I/O, so that a
    # ValidationError never leaves an orphaned file on disk.
    paper_id = _make_paper_id(filename)

    raw = {
        "paper_id":  paper_id,
        "source":    "local",
        "title":     Path(filename).stem,   # replaced by extractor after processing
        "abstract":  "",
        "authors":   [],
        "url":       "",
        "file_path": "",                    # set below after successful save
        "topic":     topic,
        "extra_metadata": {
            "original_filename": filename,
            "file_size_bytes":   len(content),
        },
    }

    # Raises ValidationError if anything is wrong — no file has been written yet.
    doc = validate(raw)

    # ── Save to DOWNLOADS_DIR/local/ ──────────────────────────────────────────
    local_dir = Path(cfg.DOWNLOADS_DIR) / "local"
    local_dir.mkdir(parents=True, exist_ok=True)

    save_path = local_dir / f"{paper_id}.pdf"
    save_path.write_bytes(content)
    logger.info("Saved uploaded file: %s (%d bytes)", save_path, len(content))

    # Update file_path now that the file is on disk
    doc.file_path = str(save_path)

    logger.info("Local upload accepted: paper_id=%s file=%s", doc.paper_id, filename)
    return doc