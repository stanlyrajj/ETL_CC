"""
pdf_renderer.py — Render LinkedIn carousel slides to a PDF using ReportLab.

Design philosophy (inspired by Anthropic "Claude in Claude" carousel):
  - Warm, editorial aesthetic: strong type hierarchy, generous white space
  - Every slide has ONE primary reading unit (a stat, a headline, a quote)
  - Accent color used sparingly — only on the single most important element
  - Cover: massive serif headline + short teaser on a toned background
  - Content slides: left accent bar + type-size contrast (large claim, small body)
  - Stat slides: the number IS the slide — enormous, centered, unavoidable
  - Quote slides: full-bleed toned background, italic serif, clean attribution
  - CTA: solid accent background, reversed-out text, one bold ask

Each slide is 1080x1080 pt (standard LinkedIn carousel).
Three schemes: light (warm cream), dark (deep navy), bold (vivid teal+yellow).
"""

import logging
import uuid
from pathlib import Path

from reportlab.lib.colors import HexColor, white, black
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth

logger = logging.getLogger(__name__)

# ── Page dimensions ───────────────────────────────────────────────────────────
W, H = 1080, 1080
PAGE_SIZE = (W, H)

# ── Color schemes ─────────────────────────────────────────────────────────────
_SCHEMES = {
    "light": {
        "bg":           HexColor("#FAF8F4"),   # warm cream
        "bg_alt":       HexColor("#F0EDE6"),   # slightly deeper cream for toned slides
        "bg_card":      HexColor("#FFFFFF"),   # pure white card
        "accent":       HexColor("#1A56DB"),   # strong blue
        "accent_light": HexColor("#E8EFFD"),   # very light blue tint
        "heading":      HexColor("#111111"),
        "body":         HexColor("#3D3D3D"),
        "muted":        HexColor("#888888"),
        "rule":         HexColor("#DDDDDD"),
        "cover_bg":     HexColor("#1A1A2E"),   # dark cover for contrast
        "cover_text":   HexColor("#FFFFFF"),
        "cover_muted":  HexColor("#A0A8C0"),
        "cta_bg":       HexColor("#1A56DB"),
        "cta_text":     HexColor("#FFFFFF"),
        "cta_muted":    HexColor("#C5D5F8"),
    },
    "dark": {
        "bg":           HexColor("#12141C"),
        "bg_alt":       HexColor("#1C1F2E"),
        "bg_card":      HexColor("#21253A"),
        "accent":       HexColor("#4ECCA3"),
        "accent_light": HexColor("#1A3A30"),
        "heading":      HexColor("#F0F0F0"),
        "body":         HexColor("#B8BCC8"),
        "muted":        HexColor("#606478"),
        "rule":         HexColor("#2A2E42"),
        "cover_bg":     HexColor("#0A0C14"),
        "cover_text":   HexColor("#FFFFFF"),
        "cover_muted":  HexColor("#6870A0"),
        "cta_bg":       HexColor("#4ECCA3"),
        "cta_text":     HexColor("#0A1A14"),
        "cta_muted":    HexColor("#1A6650"),
    },
    "bold": {
        "bg":           HexColor("#FFFFFF"),
        "bg_alt":       HexColor("#FFF8F0"),
        "bg_card":      HexColor("#FFFFFF"),
        "accent":       HexColor("#E84855"),
        "accent_light": HexColor("#FDEAEA"),
        "heading":      HexColor("#111111"),
        "body":         HexColor("#333333"),
        "muted":        HexColor("#888888"),
        "rule":         HexColor("#EEEEEE"),
        "cover_bg":     HexColor("#E84855"),
        "cover_text":   HexColor("#FFFFFF"),
        "cover_muted":  HexColor("#F9B0B5"),
        "cta_bg":       HexColor("#111111"),
        "cta_text":     HexColor("#FFFFFF"),
        "cta_muted":    HexColor("#666666"),
    },
}

_DEFAULT_SCHEME = "light"

# ── Typography ────────────────────────────────────────────────────────────────
SERIF_BOLD    = "Times-Bold"
SERIF_ITALIC  = "Times-Italic"
SERIF_ROMAN   = "Times-Roman"
SANS_BOLD     = "Helvetica-Bold"
SANS_REGULAR  = "Helvetica"
SANS_OBLIQUE  = "Helvetica-Oblique"

MARGIN   = 88    # outer margin
MARGIN_S = 64    # smaller margin for some elements
LINE_H   = 1.45  # line height multiplier


# ── Utilities ─────────────────────────────────────────────────────────────────

def _escape(text: str) -> str:
    """Normalise unicode to latin-1 for ReportLab built-in fonts."""
    if not text:
        return ""
    subs = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--",
        "\u2026": "...", "\u00a0": " ",
        "\u2022": "*",  "\u00b7": ".",
    }
    for src, dst in subs.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _wrap(text: str, font: str, size: float, max_w: float) -> list[str]:
    """Word-wrap text to fit max_w. Returns list of line strings."""
    words   = _escape(text).split()
    lines   = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        if stringWidth(candidate, font, size) <= max_w:
            current = candidate
        else:
            if current:
                lines.append(current)
            # Handle a single word longer than max_w — just add it
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _text_block(
    c: canvas.Canvas,
    text: str,
    x: float, y: float,
    font: str, size: float, color,
    max_w: float,
    max_lines: int = 20,
    align: str = "left",   # "left" | "center"
    line_h_mult: float = LINE_H,
) -> float:
    """Draw wrapped text. Returns y after last line."""
    lines  = _wrap(text, font, size, max_w)[:max_lines]
    c.setFont(font, size)
    c.setFillColor(color)
    lh = size * line_h_mult
    for line in lines:
        if align == "center":
            lw = stringWidth(line, font, size)
            c.drawString(x + (max_w - lw) / 2, y, line)
        else:
            c.drawString(x, y, line)
        y -= lh
    return y


def _rect(c, x, y, w, h, color, radius=0, alpha=1.0):
    c.setFillAlpha(alpha)
    c.setFillColor(color)
    if radius:
        c.roundRect(x, y, w, h, radius, fill=1, stroke=0)
    else:
        c.rect(x, y, w, h, fill=1, stroke=0)
    c.setFillAlpha(1.0)


def _rule(c, x, y, w, color, thickness=2):
    c.setStrokeColor(color)
    c.setLineWidth(thickness)
    c.line(x, y, x + w, y)


def _slide_counter(c, index: int, total: int, colors: dict):
    """Draw a row of dots at the bottom showing slide position."""
    dot_r   = 5
    gap     = 18
    total_w = total * (2 * dot_r) + (total - 1) * (gap - 2 * dot_r)
    x_start = (W - total_w) / 2
    y       = 42
    for i in range(total):
        if i == index:
            c.setFillColor(colors["accent"])
        else:
            c.setFillColor(colors["muted"])
            c.setFillAlpha(0.4)
        c.circle(x_start + i * gap, y, dot_r, fill=1, stroke=0)
        c.setFillAlpha(1.0)


def _left_accent_bar(c, colors, x=MARGIN - 28, y_bottom=MARGIN, height=None):
    """Vertical accent bar on the left edge — the editorial signature."""
    if height is None:
        height = H - 2 * MARGIN
    _rect(c, x, y_bottom, 6, height, colors["accent"])


def _type_tag(c, label: str, x: float, y: float, colors: dict) -> None:
    """Small pill label showing slide type (FINDING / METHOD / STAT etc.)."""
    tag   = _escape(label.upper())
    font  = SANS_BOLD
    size  = 18
    pad_x = 14
    pad_y = 8
    tw    = stringWidth(tag, font, size)
    bw    = tw + 2 * pad_x
    bh    = size + 2 * pad_y
    _rect(c, x, y - bh + pad_y, bw, bh, colors["accent_light"], radius=6)
    c.setFont(font, size)
    c.setFillColor(colors["accent"])
    c.drawString(x + pad_x, y - size + pad_y / 2, tag)


# ── Slide renderers ───────────────────────────────────────────────────────────

def _render_cover(c, slide, colors, index, total):
    """
    Cover: dark full-bleed background, massive serif title,
    short teaser below a ruled line, slide counter at bottom.
    Inspired by: bold editorial magazine covers.
    """
    # Full-bleed dark background
    _rect(c, 0, 0, W, H, colors["cover_bg"])

    # Subtle top accent stripe
    _rect(c, 0, H - 8, W, 8, colors["accent"])

    # Bottom gradient-like band (just a slightly lighter rect)
    _rect(c, 0, 0, W, 180, colors["accent"], alpha=0.08)

    title = slide.get("title", "")
    body  = slide.get("body",  "")

    usable = W - 2 * MARGIN

    # Large serif heading — the entire slide IS this
    font_size = 96
    # Shrink if title is long
    if len(title) > 30:
        font_size = 78
    if len(title) > 50:
        font_size = 64

    y = H * 0.62
    y = _text_block(c, title, MARGIN, y, SERIF_BOLD, font_size,
                    colors["cover_text"], usable, max_lines=3)

    # Ruled line
    y -= 22
    _rule(c, MARGIN, y, 100, colors["accent"], thickness=4)
    y -= 32

    # Teaser in muted smaller type
    _text_block(c, body, MARGIN, y, SERIF_ROMAN, 34,
                colors["cover_muted"], usable, max_lines=3)

    # Slide counter
    _slide_counter(c, index, total, {**colors,
                    "accent": colors["accent"],
                    "muted": colors["cover_muted"]})


def _render_finding(c, slide, colors, index, total):
    """
    Finding: warm background, left accent bar, large bold claim at top,
    smaller supporting body below, type tag, slide counter.
    """
    _rect(c, 0, 0, W, H, colors["bg"])
    _left_accent_bar(c, colors)

    title = slide.get("title", "")
    body  = slide.get("body",  "")

    usable = W - 2 * MARGIN

    # Type tag
    _type_tag(c, "Finding", MARGIN, H - MARGIN - 4, colors)

    # Main claim — big, bold, serif
    y = H - MARGIN - 80
    font_size = 68
    if len(title) > 40:
        font_size = 54
    y = _text_block(c, title, MARGIN, y, SERIF_BOLD, font_size,
                    colors["heading"], usable, max_lines=3)

    # Thin rule
    y -= 28
    _rule(c, MARGIN, y, usable * 0.25, colors["rule"], thickness=1.5)
    y -= 28

    # Supporting body text
    _text_block(c, body, MARGIN, y, SANS_REGULAR, 32,
                colors["body"], usable, max_lines=7)

    _slide_counter(c, index, total, colors)


def _render_method(c, slide, colors, index, total):
    """
    Method: card-in-background layout. White card floats on the toned bg.
    Title above the card, steps/body inside.
    """
    _rect(c, 0, 0, W, H, colors["bg_alt"])
    _left_accent_bar(c, colors, height=H - 2 * MARGIN - 20)

    title = slide.get("title", "")
    body  = slide.get("body",  "")

    usable     = W - 2 * MARGIN
    card_x     = MARGIN
    card_y     = MARGIN + 40
    card_w     = W - 2 * MARGIN
    card_h     = H - 2 * MARGIN - 100
    inner_pad  = 48

    # Type tag
    _type_tag(c, "Method", MARGIN, H - MARGIN - 4, colors)

    # Title above the card
    _text_block(c, title, MARGIN, H - MARGIN - 76, SERIF_BOLD, 58,
                colors["heading"], usable, max_lines=2)

    # White card
    _rect(c, card_x, card_y, card_w, card_h - 50, colors["bg_card"], radius=16)

    # Body inside card
    _text_block(c, body,
                card_x + inner_pad,
                card_y + card_h - 50 - inner_pad,
                SANS_REGULAR, 30, colors["body"],
                card_w - 2 * inner_pad, max_lines=10)

    _slide_counter(c, index, total, colors)


def _render_stat(c, slide, colors, index, total):
    """
    Stat: the number dominates. Extract the first numeric token and render
    it at 200pt. Everything else is secondary supporting context.
    """
    _rect(c, 0, 0, W, H, colors["bg"])

    # Accent band — top quarter
    _rect(c, 0, H * 0.65, W, H * 0.35, colors["accent_light"])
    _left_accent_bar(c, colors)

    title = slide.get("title", "")
    body  = slide.get("body",  "")

    # Try to pull a numeric/stat token from the title for big display
    import re
    stat_match = re.search(r"[\d,\.]+\s*[%xX×KkMmBbTt]?|\d+", title)
    big_stat   = stat_match.group(0).strip() if stat_match else ""
    remainder  = title.replace(big_stat, "").strip(" .,:-") if big_stat else title

    usable = W - 2 * MARGIN

    _type_tag(c, "Stat", MARGIN, H - MARGIN - 4, colors)

    if big_stat:
        # Enormous centered stat
        stat_size = 180
        if len(big_stat) > 6:
            stat_size = 130
        sw = stringWidth(big_stat, SERIF_BOLD, stat_size)
        c.setFont(SERIF_BOLD, stat_size)
        c.setFillColor(colors["accent"])
        c.drawString((W - sw) / 2, H * 0.46, big_stat)

        # Remainder label beneath stat
        if remainder:
            y = H * 0.38
            _text_block(c, remainder, MARGIN, y, SANS_BOLD, 32,
                        colors["heading"], usable,
                        max_lines=2, align="center")
    else:
        # No extractable number — fall back to large heading
        y = H * 0.62
        _text_block(c, title, MARGIN, y, SERIF_BOLD, 64,
                    colors["heading"], usable, max_lines=3)

    # Supporting body text, lower third
    if body:
        _text_block(c, body, MARGIN, H * 0.26, SANS_REGULAR, 28,
                    colors["body"], usable, max_lines=4)

    _slide_counter(c, index, total, colors)


def _render_quote(c, slide, colors, index, total):
    """
    Quote: full alternate-toned background, large italic serif quote,
    decorative opening mark, clean attribution line.
    """
    _rect(c, 0, 0, W, H, colors["bg_alt"])

    # Faint large quote mark as visual texture
    c.setFont(SERIF_BOLD, 320)
    c.setFillColor(colors["accent"])
    c.setFillAlpha(0.07)
    c.drawString(MARGIN - 30, H * 0.78, "\u201c")
    c.setFillAlpha(1.0)

    # Accent bar — horizontal this time, left edge
    _rect(c, MARGIN - 28, H * 0.18, 6, H * 0.64, colors["accent"])

    body  = slide.get("body",  "")
    title = slide.get("title", "")   # attribution

    usable = W - 2 * MARGIN

    # Quote text — large italic serif
    font_size = 52
    if len(body) > 120:
        font_size = 42
    if len(body) > 200:
        font_size = 34

    y = H * 0.66
    y = _text_block(c, body, MARGIN, y, SERIF_ITALIC, font_size,
                    colors["heading"], usable, max_lines=8)

    # Attribution
    if title:
        y -= 28
        _rule(c, MARGIN, y, 60, colors["muted"], thickness=1)
        y -= 22
        _text_block(c, f"-- {title}", MARGIN, y, SANS_REGULAR, 26,
                    colors["muted"], usable, max_lines=1)

    _slide_counter(c, index, total, colors)


def _render_cta(c, slide, colors, index, total):
    """
    CTA: solid accent background — a deliberate tonal break from all
    preceding slides. Reversed-out text. Bold singular ask.
    """
    _rect(c, 0, 0, W, H, colors["cta_bg"])

    # Texture: faint large arrow pointing right
    c.setFont(SANS_BOLD, 380)
    c.setFillColor(white)
    c.setFillAlpha(0.04)
    c.drawString(W * 0.25, H * 0.42, ">")
    c.setFillAlpha(1.0)

    title = slide.get("title", "")
    body  = slide.get("body",  "")

    usable = W - 2 * MARGIN

    # Small label at top
    c.setFont(SANS_BOLD, 20)
    c.setFillColor(colors["cta_muted"])
    c.drawString(MARGIN, H - MARGIN - 10, "WHAT TO DO NEXT")

    # Ruled line under label
    y = H - MARGIN - 30
    _rule(c, MARGIN, y, usable, colors["cta_muted"], thickness=1)
    y -= 50

    # Main CTA heading — large, reversed-out
    font_size = 80
    if len(title) > 25:
        font_size = 64
    if len(title) > 40:
        font_size = 52

    y = _text_block(c, title, MARGIN, y, SERIF_BOLD, font_size,
                    colors["cta_text"], usable, max_lines=3)
    y -= 36

    # Body
    _text_block(c, body, MARGIN, y, SANS_REGULAR, 32,
                colors["cta_muted"], usable, max_lines=4)

    _slide_counter(c, index, total,
                   {**colors, "accent": white, "muted": colors["cta_muted"]})


_RENDERERS = {
    "cover":   _render_cover,
    "finding": _render_finding,
    "method":  _render_method,
    "stat":    _render_stat,
    "quote":   _render_quote,
    "cta":     _render_cta,
}


# ── Public API ────────────────────────────────────────────────────────────────

def render(slides: list[dict], color_scheme: str, output_dir: str) -> str:
    """
    Render carousel slides to a PDF. Returns the full output path.

    Parameters
    ----------
    slides       : list of dicts with keys: type, title, body, slide_note
    color_scheme : "light" | "dark" | "bold"
    output_dir   : directory to write the PDF into
    """
    if not slides:
        raise ValueError("Cannot render a carousel with no slides.")

    scheme  = color_scheme.lower()
    colors  = _SCHEMES.get(scheme, _SCHEMES[_DEFAULT_SCHEME])
    if scheme not in _SCHEMES:
        logger.warning("Unknown color scheme %r — using %r", color_scheme, _DEFAULT_SCHEME)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"carousel_{uuid.uuid4().hex[:8]}.pdf"
    out_path = str(Path(output_dir) / filename)

    c = canvas.Canvas(out_path, pagesize=PAGE_SIZE)
    c.setTitle("ResearchRAG Carousel")

    total = len(slides)
    for i, slide in enumerate(slides):
        slide_type = slide.get("type", "finding").lower()
        renderer   = _RENDERERS.get(slide_type, _render_finding)
        renderer(c, slide, colors, i, total)
        c.showPage()

    c.save()
    logger.info("Carousel PDF: %s  slides=%d  scheme=%s", out_path, total, color_scheme)
    return out_path