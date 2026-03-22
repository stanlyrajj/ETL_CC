"""
extractor.py — Extracts text and structure from documents using Docling.

Supports PDF files and PubMed BioC JSON files.
Docling is synchronous, so extraction runs in run_in_executor.

Three-pass PDF strategy:
  Pass 1 (fast):    Docling with do_ocr=False — reads the text layer directly.
                    Governed by DOCLING_FAST_TIMEOUT seconds (default 60).
                    Accepted if it completes in time AND yields >= 200 chars.
  Pass 2 (PyMuPDF): Triggered when Pass 1 times out OR yields < 200 chars.
                    Extracts raw text from the PDF text layer in < 1 second.
                    Accepted if it yields >= 200 chars.
  Pass 3 (OCR):     Triggered only when both Pass 1 and Pass 2 fail to yield
                    enough content — indicates a scanned/image-only PDF.
                    Runs Docling with do_ocr=True. May take minutes on CPU.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from config import cfg
from ingestion.validator import DocumentInput

logger = logging.getLogger(__name__)

_MIN_CONTENT_CHARS      = 100   # minimum to consider extraction successful
_OCR_FALLBACK_THRESHOLD = 200   # passes yielding fewer chars trigger next fallback


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


def _sections_from_docling_doc(doc) -> list[Section]:
    """
    Extract Section objects from a Docling document object.
    Tries structured text items first, falls back to markdown/text export.
    """
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
        logger.warning("Docling structured iteration failed: %s", exc)
        sections = []

    # ── Fallback: export to markdown or plain text ────────────────────────────
    if not sections:
        exported_text = ""
        for export_method in ("export_to_markdown", "export_to_text"):
            if hasattr(doc, export_method):
                try:
                    exported_text = getattr(doc, export_method)()
                    if exported_text and exported_text.strip():
                        logger.info(
                            "Docling %s fallback used (%d chars)",
                            export_method, len(exported_text),
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


def _docling_fast_sync(file_path: str) -> list[Section]:
    """
    Pass 1 — Docling with OCR and table structure disabled.
    Reads the embedded text layer only. Blocking — runs in executor.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    options = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=options)}
    )
    result = converter.convert(file_path)
    return _sections_from_docling_doc(result.document)


def _pymupdf_sync(file_path: str) -> list[Section]:
    """
    Pass 2 — PyMuPDF fallback. Extracts raw text from the PDF text layer.
    Much faster than Docling (~1s) but produces no structural labels.
    Blocking — runs in executor.

    Requires: pip install pymupdf
    """
    import fitz  # pymupdf

    sections: list[Section] = []
    doc = fitz.open(file_path)

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if not text:
            continue
        # Split each page into paragraphs so chunk sizes stay manageable
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paragraphs:
            sections.append(Section(
                section_type="body",
                content=para,
                section_order=page_num,
                title="",
            ))

    doc.close()
    return sections


def _docling_ocr_sync(file_path: str) -> list[Section]:
    """
    Pass 3 — Docling with full OCR enabled.
    Only used for scanned/image-only PDFs where both Pass 1 and Pass 2 failed.
    Blocking — runs in executor. May take several minutes on CPU.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    options = PdfPipelineOptions(do_ocr=True, do_table_structure=False)
    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=options)}
    )
    result = converter.convert(file_path)
    return _sections_from_docling_doc(result.document)


async def _extract_pdf(file_path: str) -> list[Section]:
    """
    Three-pass PDF extraction with timeout and PyMuPDF fallback.

    Pass 1: Docling fast (no OCR), timeout = DOCLING_FAST_TIMEOUT seconds.
            Accepted if it completes in time AND yields >= _OCR_FALLBACK_THRESHOLD chars.
    Pass 2: PyMuPDF — triggered on timeout OR insufficient chars from Pass 1.
            Accepted if it yields >= _OCR_FALLBACK_THRESHOLD chars.
    Pass 3: Docling OCR — triggered only when both Pass 1 and Pass 2 are insufficient.
            Last resort for genuinely scanned PDFs.
    """
    loop    = asyncio.get_running_loop()
    timeout = float(cfg.DOCLING_FAST_TIMEOUT)

    # ── Pass 1: Docling fast ──────────────────────────────────────────────────
    logger.info("Extraction pass 1 (Docling fast, timeout=%ds): %s", timeout, file_path)
    sections: list[Section] = []

    try:
        sections = await asyncio.wait_for(
            loop.run_in_executor(None, _docling_fast_sync, file_path),
            timeout=timeout,
        )
        fast_chars = sum(len(s.content) for s in sections)
        logger.info("Pass 1 complete: %d chars from %s", fast_chars, file_path)

        if fast_chars >= _OCR_FALLBACK_THRESHOLD:
            return sections   # ✓ fast path succeeded

        logger.warning(
            "Pass 1 yielded only %d chars (threshold=%d) for %s — trying PyMuPDF.",
            fast_chars, _OCR_FALLBACK_THRESHOLD, file_path,
        )

    except asyncio.TimeoutError:
        logger.warning(
            "Pass 1 timed out after %ds for %s — trying PyMuPDF.",
            timeout, file_path,
        )
    except Exception as exc:
        logger.warning(
            "Pass 1 (Docling fast) failed for %s: %s — trying PyMuPDF.",
            file_path, exc,
        )

    # ── Pass 2: PyMuPDF ───────────────────────────────────────────────────────
    logger.info("Extraction pass 2 (PyMuPDF): %s", file_path)
    try:
        sections = await loop.run_in_executor(None, _pymupdf_sync, file_path)
        pymupdf_chars = sum(len(s.content) for s in sections)
        logger.info("Pass 2 complete: %d chars from %s", pymupdf_chars, file_path)

        if pymupdf_chars >= _OCR_FALLBACK_THRESHOLD:
            return sections   # ✓ PyMuPDF succeeded

        logger.warning(
            "Pass 2 (PyMuPDF) yielded only %d chars for %s — falling back to OCR.",
            pymupdf_chars, file_path,
        )

    except ImportError:
        logger.warning(
            "PyMuPDF is not installed — skipping pass 2. "
            "Install it with: pip install pymupdf"
        )
    except Exception as exc:
        logger.warning(
            "Pass 2 (PyMuPDF) failed for %s: %s — falling back to OCR.",
            file_path, exc,
        )

    # ── Pass 3: Docling OCR ───────────────────────────────────────────────────
    logger.warning(
        "Extraction pass 3 (Docling OCR) for %s — "
        "PDF may be scanned. This may take several minutes on CPU.",
        file_path,
    )
    try:
        sections = await loop.run_in_executor(None, _docling_ocr_sync, file_path)
        ocr_chars = sum(len(s.content) for s in sections)
        logger.info("Pass 3 (OCR) complete: %d chars from %s", ocr_chars, file_path)
        return sections

    except Exception as exc:
        raise ExtractionError(
            f"All extraction passes failed for {file_path!r}. "
            f"Last error (Docling OCR): {exc}"
        ) from exc


def _extract_bioc_sync(file_path: str) -> list[Section]:
    """
    Parse a PubMed BioC JSON file and return a list of Sections.
    Blocking — runs in executor.
    """
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
    Raises ExtractionError if the result contains less than _MIN_CONTENT_CHARS.
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
    loop   = asyncio.get_running_loop()

    logger.info("Extracting %s (source=%s)", path.name, document.source)

    if suffix == ".pdf":
        sections = await _extract_pdf(file_path)
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
            "If this is a PDF, ensure Docling models are downloaded by running: "
            "python -c \"from docling.document_converter import DocumentConverter; "
            "DocumentConverter()\""
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