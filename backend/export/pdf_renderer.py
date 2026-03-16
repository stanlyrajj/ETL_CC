"""
pdf_renderer.py — Render LinkedIn carousel slides to a PDF using ReportLab.

Each slide becomes one 1080x1080 point page (standard LinkedIn carousel size).
Three color schemes: dark, light, bold.
Clean typography with visual hierarchy by slide type.
"""

import logging
import uuid
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

logger = logging.getLogger(__name__)

# ── Page dimensions ───────────────────────────────────────────────────────────
PAGE_SIZE = (1080, 1080)   # 1080x1080 points — LinkedIn carousel standard

# ── Color schemes ─────────────────────────────────────────────────────────────
_SCHEMES = {
    "dark": {
        "background": HexColor("#1a1a2e"),
        "text":       HexColor("#e0e0e0"),
        "heading":    HexColor("#ffffff"),
        "accent":     HexColor("#4ecca3"),
        "muted":      HexColor("#a0a0b0"),
    },
    "light": {
        "background": HexColor("#ffffff"),
        "text":       HexColor("#333333"),
        "heading":    HexColor("#111111"),
        "accent":     HexColor("#0077b5"),   # LinkedIn blue
        "muted":      HexColor("#666666"),
    },
    "bold": {
        "background": HexColor("#ff6b35"),
        "text":       HexColor("#ffffff"),
        "heading":    HexColor("#ffffff"),
        "accent":     HexColor("#ffee00"),
        "muted":      HexColor("#ffe0d0"),
    },
}

_DEFAULT_SCHEME = "light"

# ── Typography ────────────────────────────────────────────────────────────────
_FONT_REGULAR = "Helvetica"
_FONT_BOLD    = "Helvetica-Bold"

_MARGIN       = 80   # points from edge
_LINE_HEIGHT  = 1.35


# ── Internal helpers ──────────────────────────────────────────────────────────

def _escape(text: str) -> str:
    """Strip characters that can cause ReportLab rendering issues."""
    if not text:
        return ""
    # Replace common problematic unicode with ASCII equivalents
    replacements = {
        "\u2018": "'", "\u2019": "'",   # curly single quotes
        "\u201c": '"', "\u201d": '"',   # curly double quotes
        "\u2013": "-", "\u2014": "--",  # en/em dash
        "\u2026": "...",                # ellipsis
        "\u00a0": " ",                  # non-breaking space
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Drop any remaining non-latin-1 characters ReportLab can't encode
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _fill_background(c: canvas.Canvas, colors: dict) -> None:
    """Fill the whole page with the background color."""
    c.setFillColor(colors["background"])
    c.rect(0, 0, PAGE_SIZE[0], PAGE_SIZE[1], fill=1, stroke=0)


def _draw_accent_bar(c: canvas.Canvas, colors: dict) -> None:
    """Draw a thin accent bar at the top of every slide."""
    bar_height = 12
    c.setFillColor(colors["accent"])
    c.rect(0, PAGE_SIZE[1] - bar_height, PAGE_SIZE[0], bar_height, fill=1, stroke=0)


def _wrap_text(text: str, font: str, size: float, max_width: float) -> list[str]:
    """
    Wrap text into lines that fit within max_width using ReportLab string width.
    Returns a list of line strings.
    """
    from reportlab.pdfbase.pdfmetrics import stringWidth
    words  = text.split()
    lines  = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        if stringWidth(candidate, font, size) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _draw_text_block(
    c:         canvas.Canvas,
    text:      str,
    x:         float,
    y:         float,
    font:      str,
    size:      float,
    color,
    max_width: float,
    max_lines: int = 12,
) -> float:
    """
    Draw wrapped text starting at (x, y), moving downward.
    Returns the y position after the last line drawn.
    """
    lines = _wrap_text(_escape(text), font, size, max_width)[:max_lines]
    c.setFont(font, size)
    c.setFillColor(color)
    line_h = size * _LINE_HEIGHT
    for line in lines:
        c.drawString(x, y, line)
        y -= line_h
    return y


# ── Slide renderers by type ───────────────────────────────────────────────────

def _render_cover(c: canvas.Canvas, slide: dict, colors: dict) -> None:
    w, h   = PAGE_SIZE
    margin = _MARGIN
    usable = w - 2 * margin

    _fill_background(c, colors)
    _draw_accent_bar(c, colors)

    # Large title, vertically centred slightly above middle
    title = slide.get("title", "")
    body  = slide.get("body",  "")

    # Title
    y = h * 0.58
    y = _draw_text_block(c, title, margin, y, _FONT_BOLD, 72, colors["heading"], usable, max_lines=3)

    # Divider line
    y -= 20
    c.setStrokeColor(colors["accent"])
    c.setLineWidth(4)
    c.line(margin, y, margin + 120, y)
    y -= 32

    # Subtitle / body
    _draw_text_block(c, body, margin, y, _FONT_REGULAR, 36, colors["muted"], usable, max_lines=4)

    # Slide number indicator dot
    c.setFillColor(colors["accent"])
    c.circle(w / 2, margin / 2, 8, fill=1, stroke=0)


def _render_content(c: canvas.Canvas, slide: dict, colors: dict) -> None:
    """Used for finding, method, stat slides."""
    w, h   = PAGE_SIZE
    margin = _MARGIN
    usable = w - 2 * margin

    _fill_background(c, colors)
    _draw_accent_bar(c, colors)

    title = slide.get("title", "")
    body  = slide.get("body",  "")
    hint  = slide.get("visual_hint", "")

    # Slide type label (small caps style)
    label = slide.get("type", "").upper()
    c.setFont(_FONT_REGULAR, 22)
    c.setFillColor(colors["accent"])
    c.drawString(margin, h - margin - 30, label)

    # Heading
    y = h - margin - 90
    y = _draw_text_block(c, title, margin, y, _FONT_BOLD, 56, colors["heading"], usable, max_lines=2)
    y -= 24

    # Body text
    y = _draw_text_block(c, body, margin, y, _FONT_REGULAR, 34, colors["text"], usable, max_lines=8)

    # Visual hint in muted text at bottom if present
    if hint:
        _draw_text_block(c, f"[{hint}]", margin, margin + 40, _FONT_REGULAR, 22, colors["muted"], usable, max_lines=1)


def _render_quote(c: canvas.Canvas, slide: dict, colors: dict) -> None:
    w, h   = PAGE_SIZE
    margin = _MARGIN
    usable = w - 2 * margin

    _fill_background(c, colors)
    _draw_accent_bar(c, colors)

    body  = slide.get("body",  "")
    title = slide.get("title", "")

    # Opening quote mark
    c.setFont(_FONT_BOLD, 160)
    c.setFillColor(colors["accent"])
    c.setFillAlpha(0.3)
    c.drawString(margin - 20, h * 0.72, "\u201c")
    c.setFillAlpha(1.0)

    # Quote text
    y = h * 0.64
    y = _draw_text_block(c, body, margin, y, _FONT_BOLD, 44, colors["heading"], usable, max_lines=6)

    # Attribution
    if title:
        y -= 20
        _draw_text_block(c, f"— {title}", margin, y, _FONT_REGULAR, 28, colors["muted"], usable, max_lines=2)


def _render_cta(c: canvas.Canvas, slide: dict, colors: dict) -> None:
    w, h   = PAGE_SIZE
    margin = _MARGIN
    usable = w - 2 * margin

    _fill_background(c, colors)
    _draw_accent_bar(c, colors)

    title = slide.get("title", "")
    body  = slide.get("body",  "")

    # Large centred CTA heading
    y = h * 0.60
    y = _draw_text_block(c, title, margin, y, _FONT_BOLD, 64, colors["accent"], usable, max_lines=2)
    y -= 28

    # Body
    _draw_text_block(c, body, margin, y, _FONT_REGULAR, 36, colors["text"], usable, max_lines=4)

    # Decorative bottom bar
    bar_w = w - 2 * margin
    c.setFillColor(colors["accent"])
    c.rect(margin, margin + 20, bar_w, 6, fill=1, stroke=0)


_SLIDE_RENDERERS = {
    "cover":   _render_cover,
    "finding": _render_content,
    "method":  _render_content,
    "stat":    _render_content,
    "quote":   _render_quote,
    "cta":     _render_cta,
}


# ── Public API ────────────────────────────────────────────────────────────────

def render(slides: list[dict], color_scheme: str, output_dir: str) -> str:
    """
    Render a list of carousel slides to a PDF file.

    Parameters
    ----------
    slides       : list of slide dicts with keys: type, title, body, visual_hint
    color_scheme : "dark" | "light" | "bold"
    output_dir   : directory to save the PDF

    Returns
    -------
    Full path to the generated PDF file.
    """
    if not slides:
        raise ValueError("Cannot render a carousel with no slides.")

    colors = _SCHEMES.get(color_scheme.lower(), _SCHEMES[_DEFAULT_SCHEME])
    if color_scheme.lower() not in _SCHEMES:
        logger.warning("Unknown color scheme %r — using %r", color_scheme, _DEFAULT_SCHEME)

    # Build output path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename  = f"carousel_{uuid.uuid4().hex[:8]}.pdf"
    out_path  = str(Path(output_dir) / filename)

    c = canvas.Canvas(out_path, pagesize=PAGE_SIZE)
    c.setTitle("ResearchRAG Carousel")

    for i, slide in enumerate(slides):
        slide_type = slide.get("type", "finding").lower()
        renderer   = _SLIDE_RENDERERS.get(slide_type, _render_content)
        renderer(c, slide, colors)
        c.showPage()

    c.save()
    logger.info("Carousel PDF saved: %s (%d slides, scheme=%s)", out_path, len(slides), color_scheme)
    return out_path