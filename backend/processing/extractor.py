"""
extractor.py — Extracts text and structure from documents using Docling.

Supports PDF files and PubMed BioC JSON files.
Docling is synchronous, so extraction runs in run_in_executor.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ingestion.validator import DocumentInput

logger = logging.getLogger(__name__)

_MIN_CONTENT_CHARS = 100


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class Section:
    section_type:  str    # abstract, body, title, reference, caption
    content:       str
    section_order: int
    title:         str = ""


@dataclass
class ExtractedDocument:
    paper_id: str
    title:    str
    abstract: str
    authors:  list[str]
    source:   str
    sections: list[Section] = field(default_factory=list)


# ── Exception ─────────────────────────────────────────────────────────────────

class ExtractionError(Exception):
    pass


# ── Internal helpers ──────────────────────────────────────────────────────────

def _label_to_section_type(label_str: str) -> str:
    """Map a Docling label string to one of our four section type names."""
    s = label_str.lower()
    if "title"    in s: return "title"
    if "abstract" in s: return "abstract"
    if "refer"    in s: return "reference"
    if "caption"  in s: return "caption"
    return "body"


def _extract_pdf_sync(file_path: str) -> list[Section]:
    """
    Run Docling on a PDF file and return a list of Sections.
    This is blocking — call it inside run_in_executor.

    Supports Docling v2. Tries structured item iteration first,
    falls back to full-document markdown export if that yields nothing.
    """
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result    = converter.convert(file_path)
    doc       = result.document

    sections: list[Section] = []
    order = 0

    # ── Primary: structured text items (Docling v2) ───────────────────────────
    try:
        items = []
        if hasattr(doc, "texts") and doc.texts:
            items = list(doc.texts)
        elif hasattr(doc, "body") and hasattr(doc.body, "children"):
            items = list(doc.body.children)

        for item in items:
            label = getattr(item, "label", None)
            text  = (getattr(item, "text", None) or "").strip()
            if not text:
                continue
            sections.append(Section(
                section_type=_label_to_section_type(str(label) if label else ""),
                content=text,
                section_order=order,
                title="",
            ))
            order += 1

    except Exception as exc:
        logger.error(
            "Docling structured iteration failed for %s: %s",
            file_path, exc, exc_info=True,
        )
        sections = []

    # ── Fallback: try multiple export methods until one yields content ─────────
    if not sections:
        exported_text = ""

        for export_method in ("export_to_markdown", "export_to_text"):
            if hasattr(doc, export_method):
                try:
                    exported_text = getattr(doc, export_method)()
                    if exported_text and exported_text.strip():
                        logger.info(
                            "%s fallback used for %s (%d chars)",
                            export_method, file_path, len(exported_text)
                        )
                        break
                except Exception as exc:
                    logger.warning("Docling %s failed: %s", export_method, exc)

        if exported_text and exported_text.strip():
            paragraphs = [p.strip() for p in exported_text.split("\n\n") if p.strip()]
            for i, para in enumerate(paragraphs):
                sections.append(Section(
                    section_type="body",
                    content=para,
                    section_order=i,
                    title="",
                ))

    return sections


def _extract_bioc_sync(file_path: str) -> list[Section]:
    """
    Parse a PubMed BioC JSON file and return a list of Sections.
    This is blocking — call it inside run_in_executor.
    """
    # FIX E1: Wrap json.load() so malformed input raises ExtractionError
    # instead of a raw json.JSONDecodeError that the pipeline cannot catch.
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ExtractionError(
            f"Invalid BioC JSON in {file_path!r}: {exc}"
        ) from exc

    sections: list[Section] = []
    order = 0

    for document in data.get("documents", []):
        for passage in document.get("passages", []):
            infons = passage.get("infons", {})
            text   = passage.get("text", "").strip()
            if not text:
                continue

            raw_type      = infons.get("type", "body").lower()
            section_title = infons.get("section_type", "")

            if "abstract" in raw_type:
                section_type = "abstract"
            elif "title" in raw_type:
                section_type = "title"
            elif "ref" in raw_type:
                section_type = "reference"
            else:
                section_type = "body"

            sections.append(Section(
                section_type=section_type,
                content=text,
                section_order=order,
                title=section_title,
            ))
            order += 1

    return sections


# ── Public API ────────────────────────────────────────────────────────────────

async def extract(document: DocumentInput) -> ExtractedDocument:
    """
    Extract text and structure from a document.

    Supports PDF files and PubMed BioC JSON files.
    Raises ExtractionError if the result contains less than 100 characters.
    """
    file_path = document.file_path

    if not file_path:
        raise ExtractionError(
            f"No file_path set for paper_id={document.paper_id!r}."
        )

    path = Path(file_path)
    if not path.exists():
        raise ExtractionError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    # FIX E3: Use get_running_loop() — get_event_loop() is deprecated in
    # Python 3.10+ and may raise RuntimeError inside coroutines in Python 3.12.
    loop = asyncio.get_running_loop()

    logger.info("Extracting %s (source=%s)", path.name, document.source)

    if suffix == ".pdf":
        sections = await loop.run_in_executor(None, _extract_pdf_sync, file_path)
    elif suffix == ".json":
        sections = await loop.run_in_executor(None, _extract_bioc_sync, file_path)
    else:
        raise ExtractionError(
            f"Unsupported file type '{suffix}' for paper_id={document.paper_id!r}. "
            "Supported types: .pdf, .json (BioC)"
        )

    total_chars = sum(len(s.content) for s in sections)
    if total_chars < _MIN_CONTENT_CHARS:
        raise ExtractionError(
            f"Extraction produced only {total_chars} characters for "
            f"paper_id={document.paper_id!r} — minimum is {_MIN_CONTENT_CHARS}. "
            "If this is a PDF, ensure Docling models are fully downloaded by running: "
            "python -c \"from docling.document_converter import DocumentConverter; DocumentConverter()\""
        )

    logger.info(
        "Extraction complete: paper_id=%s sections=%d total_chars=%d",
        document.paper_id, len(sections), total_chars,
    )

    return ExtractedDocument(
        paper_id=document.paper_id,
        title=document.title,
        abstract=document.abstract,
        authors=document.authors,
        source=document.source,
        sections=sections,
    )