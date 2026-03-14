"""
carousel_renderer.py — Renders carousel slides as PNG + PDF

"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas

from config import cfg

logger = logging.getLogger(__name__)

SLIDE_W = 1080
SLIDE_H = 1080

# ── Emerald palette presets ──────────────────────────────────────────────────

EMERALD_PRESETS = {
    "emerald_dark": {
        "bg_color":       "#0A1628",
        "accent_color":   "#00C896",
        "text_color":     "#E8F5F0",
        "secondary_color":"#7ECDB8",
        "card_color":     "#0F2040",
        "font":           "playfair",
        "accent_shape":   "circle",
        "footer_text":    "ResearchRAG",
    },
    "emerald_light": {
        "bg_color":       "#F0FAF6",
        "accent_color":   "#00916E",
        "text_color":     "#0A2E1E",
        "secondary_color":"#3D8B72",
        "card_color":     "#D6F0E8",
        "font":           "lora",
        "accent_shape":   "line",
        "footer_text":    "ResearchRAG",
    },
    "emerald_bold": {
        "bg_color":       "#011A10",
        "accent_color":   "#00FF9D",
        "text_color":     "#FFFFFF",
        "secondary_color":"#80FFCE",
        "card_color":     "#012A1A",
        "font":           "sans",
        "accent_shape":   "circle",
        "footer_text":    "ResearchRAG",
    },
}

DEFAULT_STYLE = EMERALD_PRESETS["emerald_dark"]

FONT_IMPORTS = {
    "playfair": "https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=DM+Mono:wght@400;500&display=swap",
    "lora":     "https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=DM+Mono:wght@400&display=swap",
    "sans":     "https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap",
    "mono":     "https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&display=swap",
}

FONT_FAMILIES = {
    "playfair": "'Playfair Display', Georgia, 'Times New Roman', serif",
    "lora":     "'Lora', Georgia, 'Times New Roman', serif",
    "sans":     "'Plus Jakarta Sans', 'Trebuchet MS', Arial, sans-serif",
    "mono":     "'DM Mono', 'Courier New', monospace",
}

FONT_FALLBACKS = {
    "playfair": "Georgia, 'Times New Roman', serif",
    "lora":     "Georgia, 'Times New Roman', serif",
    "sans":     "'Trebuchet MS', Arial, sans-serif",
    "mono":     "'Courier New', monospace",
}


# SEC-07: Full HTML escaping including quotes — prevents XSS in Playwright HTML.
# Old: only escaped &, <, > — missed quotes, broke out of HTML attributes.
def _esc(text: str) -> str:
    """Full HTML escaping including single and double quotes (XSS-safe)."""
    import html as _html
    return _html.escape(str(text or ""), quote=True)


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_slide_html(slide: Dict, style: Dict, slide_num: int, total: int) -> str:
    """Generate self-contained HTML for one slide."""
    font_key  = style.get("font", "playfair")
    font_url  = FONT_IMPORTS.get(font_key, FONT_IMPORTS["playfair"])
    font_fam  = FONT_FAMILIES.get(font_key, FONT_FAMILIES["playfair"])
    font_fall = FONT_FALLBACKS.get(font_key, "Georgia, serif")

    bg     = style.get("bg_color", "#0A1628")
    accent = style.get("accent_color", "#00C896")
    text   = style.get("text_color", "#E8F5F0")
    sec    = style.get("secondary_color", "#7ECDB8")
    card   = style.get("card_color", "#0F2040")
    footer = style.get("footer_text", "ResearchRAG")
    shape  = style.get("accent_shape", "circle")

    ad = _hex_rgba(accent, 0.15)
    am = _hex_rgba(accent, 0.35)

    stype     = slide.get("slide_type", "finding")
    title     = slide.get("title", "")
    subtitle  = slide.get("subtitle", "")
    bullets   = slide.get("bullets", [])
    stat      = slide.get("stat", "")
    stat_label = slide.get("stat_label", "")
    quote     = slide.get("quote", "")

    if stype == "cover":
        content = f"""
        <div class="cover-badge">&#9675; Research Paper</div>
        <div class="title-xl">{_esc(title)}</div>
        <div class="rule"></div>
        <div class="subtitle">{_esc(subtitle)}</div>
        <div class="authors">{_esc(stat)}</div>
        <div class="spacer"></div>
        """
    elif stype == "stat":
        content = f"""
        <div class="label">Key Finding</div>
        <div class="title">{_esc(title)}</div>
        <div class="rule"></div>
        <div class="stat-wrap">
            <div class="stat-num">{_esc(stat)}</div>
            <div class="stat-lbl">{_esc(stat_label)}</div>
        </div>
        """
    elif stype == "quote":
        content = f"""
        <div class="label">{_esc(title)}</div>
        <div class="spacer"></div>
        <div class="quote-block">
            <div class="qmark">&ldquo;</div>
            <div class="qtext">{_esc(quote)}</div>
        </div>
        <div class="spacer"></div>
        """
    elif stype == "cta":
        bullets_html = "".join(
            f'<div class="cta-item">&#8594; {_esc(b)}</div>' for b in bullets
        )
        content = f"""
        <div class="label">Take Away</div>
        <div class="title">{_esc(title)}</div>
        <div class="rule"></div>
        <div class="cta-list">{bullets_html}</div>
        """
    else:
        bullets_html = "".join(
            f'<div class="bullet"><span class="bdot">&#9675;</span><span>{_esc(b)}</span></div>'
            for b in bullets
        )
        content = f"""
        <div class="label">{_esc(stype.replace("_", " ").title())}</div>
        <div class="title">{_esc(title)}</div>
        <div class="rule"></div>
        <div class="bullets">{bullets_html}</div>
        """

    if shape == "circle":
        shape_css = f"""
        .shape1 {{
            position:absolute; width:340px; height:340px; border-radius:50%;
            border:2px solid {am}; top:-80px; right:-80px; pointer-events:none;
        }}
        .shape2 {{
            position:absolute; width:160px; height:160px; border-radius:50%;
            background:{ad}; bottom:60px; left:-40px; pointer-events:none;
        }}
        """
    elif shape == "line":
        shape_css = f"""
        .shape1 {{
            position:absolute; width:4px; height:100%;
            background:linear-gradient({accent}, transparent);
            left:0; top:0; pointer-events:none;
        }}
        .shape2 {{ display:none; }}
        """
    else:
        shape_css = ".shape1, .shape2 { display:none; }"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="{font_url}" rel="stylesheet">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    width:{SLIDE_W}px; height:{SLIDE_H}px; overflow:hidden;
    background:{bg}; color:{text};
    font-family:{font_fam},{font_fall};
    position:relative;
}}
.slide {{
    width:{SLIDE_W}px; height:{SLIDE_H}px;
    display:flex; flex-direction:column;
    padding:72px; position:relative; overflow:hidden;
}}
{shape_css}
.footer {{
    position:absolute; bottom:40px; left:72px; right:72px;
    display:flex; align-items:center; gap:12px;
    font-family:'DM Mono',monospace; font-size:22px;
    color:{sec};
}}
.footer-brand {{ color:{accent}; font-weight:500; }}
.spacer {{ flex:1; }}
.label {{
    font-family:'DM Mono',monospace; font-size:20px;
    letter-spacing:.14em; text-transform:uppercase;
    color:{accent}; margin-bottom:24px;
}}
.title {{
    font-size:72px; font-weight:700; line-height:1.12;
    letter-spacing:-.02em; color:{text}; margin-bottom:28px;
}}
.title-xl {{
    font-size:88px; font-weight:700; line-height:1.08;
    letter-spacing:-.03em; color:{text};
}}
.subtitle {{ font-size:32px; color:{sec}; line-height:1.5; margin-bottom:48px; }}
.authors {{ font-family:'DM Mono',monospace; font-size:24px; color:{sec}; margin-top:16px; }}
.rule {{ width:80px; height:4px; background:{accent}; border-radius:2px; margin-bottom:48px; }}
.cover-badge {{
    display:inline-flex; align-items:center; font-family:'DM Mono',monospace;
    font-size:20px; letter-spacing:.1em; text-transform:uppercase;
    color:{accent}; background:{ad}; border:1px solid {am};
    padding:8px 18px; border-radius:99px; margin-bottom:36px; width:fit-content;
}}
.bullets {{ display:flex; flex-direction:column; gap:20px; flex:1; }}
.bullet {{
    display:flex; align-items:flex-start; gap:20px;
    background:{ad}; border:1px solid {am};
    border-radius:12px; padding:24px 28px;
    font-size:30px; line-height:1.5; color:{text};
}}
.bdot {{ color:{accent}; font-size:18px; margin-top:6px; flex-shrink:0; }}
.stat-wrap {{ text-align:center; flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:16px; }}
.stat-num {{ font-size:160px; font-weight:700; line-height:1; color:{accent}; letter-spacing:-.04em; }}
.stat-lbl {{ font-size:36px; color:{sec}; text-align:center; max-width:600px; line-height:1.4; }}
.quote-block {{ flex:1; display:flex; flex-direction:column; justify-content:center; padding:40px; background:{ad}; border-left:6px solid {accent}; border-radius:0 16px 16px 0; }}
.qmark {{ font-size:120px; line-height:.7; color:{accent}; opacity:.4; font-family:Georgia,serif; margin-bottom:16px; }}
.qtext {{ font-size:36px; line-height:1.6; color:{text}; font-style:italic; }}
.cta-list {{ display:flex; flex-direction:column; gap:24px; flex:1; justify-content:center; }}
.cta-item {{ font-family:'DM Mono',monospace; font-size:28px; letter-spacing:.05em; color:{accent}; border:1px solid {am}; padding:20px 36px; border-radius:8px; }}
</style>
</head>
<body>
<div class="slide">
    <div class="shape1"></div>
    <div class="shape2"></div>
    {content}
    <div class="footer">
        <span class="footer-brand">{_esc(footer)}</span>
        <span style="opacity:.4"> &middot; </span>
        <span>{slide_num}/{total}</span>
    </div>
</div>
</body>
</html>"""


async def render_carousel(
    paper_id: str,
    slides: List[Dict],
    style: Optional[Dict] = None,
) -> Tuple[str, List[str]]:
    """
    Render slides to PNG, then assemble into PDF.

    PERF-04: asyncio.Semaphore(3) limits peak concurrent Playwright pages to 3,
             preventing OOM on large carousels (~75MB per page).
             finally block in render_one() guarantees page.close() on any exit path.
    """
    if not slides:
        raise ValueError("No slides to render.")

    merged_style = {**DEFAULT_STYLE, **(style or {})}
    total = len(slides)

    out_dir = Path(cfg.CAROUSEL_OUTPUT_DIR) / paper_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise ImportError(
            "Playwright not installed.\n"
            "Run: pip install playwright && playwright install chromium"
        )

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()

        # PERF-04: max 3 concurrent browser pages — prevents OOM on large carousels
        _page_sem = asyncio.Semaphore(3)

        async def render_one(slide: Dict, idx: int) -> str:
            async with _page_sem:
                page = await browser.new_page(viewport={"width": SLIDE_W, "height": SLIDE_H})
                try:
                    html = build_slide_html(slide, merged_style, idx, total)
                    await page.set_content(html, wait_until="networkidle")
                    await asyncio.sleep(0.6)  # let fonts load
                    fname = str(out_dir / f"slide_{idx:02d}.png")
                    await page.screenshot(path=fname, full_page=False)
                    return fname
                finally:
                    await page.close()  # guaranteed cleanup even on error

        tasks = [render_one(slide, i + 1) for i, slide in enumerate(slides)]
        png_paths = await asyncio.gather(*tasks)
        await browser.close()

    logger.info(f"[{paper_id}] Rendered {len(png_paths)} slides.")

    pdf_path = str(out_dir / f"{paper_id}_carousel.pdf")
    from reportlab.lib.pagesizes import landscape
    c = rl_canvas.Canvas(pdf_path, pagesize=landscape((SLIDE_W, SLIDE_H)))
    for png in png_paths:
        img = ImageReader(png)
        c.drawImage(img, 0, 0, width=SLIDE_W, height=SLIDE_H)
        c.showPage()
    c.save()
    logger.info(f"[{paper_id}] PDF assembled → {pdf_path}")

    return pdf_path, list(png_paths)