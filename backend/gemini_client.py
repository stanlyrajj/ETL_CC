"""
gemini_client.py — Google Gemini API wrapper (free tier via Google AI Studio).

Replaces all Anthropic/Claude API calls.
Uses gemini-1.5-flash (free tier, 1M token context window).

Install:
    pip install google-generativeai

Get free API key:
    https://aistudio.google.com/app/apikey
"""

import logging
import time
from typing import List, Dict, Optional, Any

import google.generativeai as genai

from config import cfg

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

TEACHING_SYSTEM = """You are an expert research educator who teaches academic papers
at different levels of understanding. You maintain a conversation about a specific paper,
remember what was discussed, and adjust your explanations to the user's level.

Always be:
- Accurate to the paper's actual content
- Clear and engaging
- Ready to answer follow-up questions
- Adaptive when the user asks to go deeper or simpler

Never make up facts not in the paper."""

TEACHING_PROMPT = """You are teaching this research paper to a {level} audience.

PAPER CONTEXT:
Title: {title}
Authors: {authors}

{paper_context}

LEVEL DEFINITIONS:
- beginner: No science background. Use everyday analogies. Avoid all jargon.
- intermediate: Some background (undergrad level). Can handle terms if briefly explained.
- advanced: Expert in the field. Focus on methodology, limitations, and open questions.

CONVERSATION SO FAR:
{history}

USER MESSAGE: {user_message}

Respond naturally as a teacher. Be conversational, not bullet-point heavy.
If this is the first message, give a welcoming introduction and ask what aspect they want to explore first.
"""

TWITTER_THREAD_SYSTEM = """You are a science communicator creating Twitter/X threads.
Threads are accurate, accessible, and spark genuine curiosity.
Respond with valid JSON only — no markdown, no preamble."""

TWITTER_THREAD_PROMPT = """Create a Twitter thread from this research paper.

Title: {title}
Authors: {authors}
Abstract: {abstract}
Key content: {key_content}

Requirements:
- 6-10 tweets
- Each tweet under 270 characters (leave room for numbering)
- Tweet 1: compelling hook
- Last tweet: key takeaway + call to action
- Number as "1/N" format
- 3-5 relevant hashtags in the last tweet only

Respond with JSON only:
{{"tweets": ["tweet 1", "tweet 2", ...], "hashtags": ["#Tag1", ...]}}"""

LINKEDIN_POST_SYSTEM = """You are a professional science writer creating LinkedIn posts.
Posts are professional yet approachable with clear real-world impact.
Respond with valid JSON only — no markdown, no preamble."""

LINKEDIN_POST_PROMPT = """Write a LinkedIn post about this research paper.

Title: {title}
Authors: {authors}
Abstract: {abstract}
Key content: {key_content}

Requirements:
- 200-350 words
- Structure: Hook → What they did → Why it matters → Key finding → Discussion question
- Professional tone
- End with an open question
- 4-6 hashtags at the end

Respond with JSON only:
{{"content": "post text without hashtags", "hashtags": ["#Tag1", ...]}}"""

CAROUSEL_SYSTEM = """You are a science communicator creating LinkedIn carousel slides.
Each slide must be standalone yet part of a cohesive narrative.
Respond with valid JSON only — no markdown, no preamble."""

CAROUSEL_PROMPT = """Create a LinkedIn carousel from this research paper. Generate 6-8 slides.

Title: {title}
Authors: {authors}
Abstract: {abstract}
Key content: {key_content}

Slide types:
- cover: title slide (title=short punchy version, subtitle=one-sentence hook, stat=authors)
- finding: key insight with 2-3 bullet points
- method: what they did (2-3 bullets)
- stat: one big number/percentage (stat=the number like "87%", stat_label=explanation)
- quote: striking sentence from paper (quote field)
- cta: final slide — 3 action bullets for the reader

Requirements:
- Slide 1 must be "cover", last must be "cta"
- Include at least one "stat" slide if numbers appear in the abstract
- Titles max 8 words, bullets max 12 words each
- No unexplained jargon

Respond with JSON only:
{{"slides": [{{"slide_type": "...", "title": "...", "subtitle": "...", "bullets": [], "stat": "", "stat_label": "", "quote": ""}}], "hashtags": ["#Tag1"]}}"""


# ── Gemini client ─────────────────────────────────────────────────────────────

class GeminiClient:
    """
    Thin wrapper around Google Generative AI SDK.
    Handles retries, response parsing, and conversation history.
    """

    def __init__(self):
        if not cfg.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not set. "
                "Get a free key at https://aistudio.google.com/app/apikey"
            )
        genai.configure(api_key=cfg.GEMINI_API_KEY)
        self._model = genai.GenerativeModel(
            model_name=cfg.GEMINI_MODEL,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048,
            ),
        )
        logger.info(f"Gemini client initialised with model={cfg.GEMINI_MODEL}")

    def _call(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048) -> str:
        """Single call with retry logic."""
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        last_exc = None
        for attempt in range(1, cfg.MAX_RETRIES + 1):
            try:
                response = self._model.generate_content(full_prompt)
                return response.text.strip()
            except Exception as exc:
                last_exc = exc
                if attempt < cfg.MAX_RETRIES:
                    wait = cfg.RETRY_BACKOFF_BASE ** (attempt - 1)
                    logger.warning(f"[Gemini] attempt {attempt} failed: {exc}. Retry in {wait}s")
                    time.sleep(wait)
        raise RuntimeError(f"Gemini API failed after {cfg.MAX_RETRIES} attempts: {last_exc}")

    def _parse_json(self, raw: str, context: str) -> Any:
        """Parse JSON from Gemini response, stripping any accidental markdown."""
        import json, re
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error(f"[Gemini] JSON parse failed for {context}: {exc}\nRaw: {raw[:300]}")
            raise ValueError(f"Gemini returned invalid JSON for {context}.") from exc

    # ── Chat / Teaching ───────────────────────────────────────────────────

    def chat_response(
        self,
        paper_context: str,
        paper_title: str,
        paper_authors: str,
        message_history: List[Dict[str, str]],
        user_message: str,
        level: str = "beginner",
    ) -> str:
        """
        Generate a teaching chat response with full conversation memory.
        message_history: list of {"role": "user"|"assistant", "content": "..."}
        """
        # Format history for prompt
        history_text = ""
        for msg in message_history[-10:]:  # Last 10 messages for context
            role = "User" if msg["role"] == "user" else "Teacher"
            history_text += f"{role}: {msg['content']}\n\n"

        prompt = TEACHING_PROMPT.format(
            level=level,
            title=paper_title,
            authors=paper_authors,
            paper_context=paper_context[:3000],
            history=history_text or "(This is the start of the conversation)",
            user_message=user_message,
        )

        return self._call(prompt, system=TEACHING_SYSTEM, max_tokens=1024)

    # ── Social content ────────────────────────────────────────────────────

    def generate_twitter_thread(
        self, title: str, authors: str, abstract: str, key_content: str
    ) -> Dict:
        prompt = TWITTER_THREAD_PROMPT.format(
            title=title, authors=authors,
            abstract=abstract[:600], key_content=key_content[:800],
        )
        raw = self._call(prompt, system=TWITTER_THREAD_SYSTEM, max_tokens=1500)
        data = self._parse_json(raw, f"twitter/{title[:30]}")

        import re as _re
        tweets = data.get("tweets", [])
        n = len(tweets)
        numbered = []
        for i, t in enumerate(tweets, 1):
            t_clean = _re.sub(r"^\d+/\d+\s*", "", t).strip()
            tweet = f"{i}/{n} {t_clean}"
            numbered.append(tweet[:280])

        return {"tweets": numbered, "hashtags": data.get("hashtags", [])}

    def generate_linkedin_post(
        self, title: str, authors: str, abstract: str, key_content: str
    ) -> Dict:
        prompt = LINKEDIN_POST_PROMPT.format(
            title=title, authors=authors,
            abstract=abstract[:600], key_content=key_content[:800],
        )
        raw = self._call(prompt, system=LINKEDIN_POST_SYSTEM, max_tokens=1500)
        return self._parse_json(raw, f"linkedin/{title[:30]}")

    def generate_carousel_content(
        self, title: str, authors: str, abstract: str, key_content: str
    ) -> Dict:
        prompt = CAROUSEL_PROMPT.format(
            title=title, authors=authors,
            abstract=abstract[:600], key_content=key_content[:800],
        )
        raw = self._call(prompt, system=CAROUSEL_SYSTEM, max_tokens=2000)
        return self._parse_json(raw, f"carousel/{title[:30]}")


# ── Singleton ─────────────────────────────────────────────────────────────────

_gemini: Optional[GeminiClient] = None

def get_gemini() -> GeminiClient:
    global _gemini
    if _gemini is None:
        _gemini = GeminiClient()
    return _gemini