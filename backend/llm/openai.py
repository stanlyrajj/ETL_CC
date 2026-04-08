"""
openai.py — OpenAI / OpenRouter / Groq implementation of LLMProvider.
"""

import asyncio
import logging

from openai import OpenAI

from config import cfg
from llm.base import LLMProvider, parse_json_response, sanitize_context
from llm.rate_limiter import llm_rate_limiter

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_CHAT_PROMPT = """You are helping a user understand the following research paper.

Paper title: {title}
Authors: {authors}

{context}

The user's knowledge level is: {level}
- beginner: use plain language, avoid jargon, use analogies
- intermediate: assume basic domain knowledge
- advanced: use technical language freely"""

_TWITTER_PROMPT = """You are creating a Twitter/X thread to share insights from an academic research paper.

Paper title: {title}
Creator intent: {description}

Inferred from the intent above, apply these attributes as you write:
- Audience: who will read this thread
- Voice: first-person researcher, third-person reporter, or explainer
- Hook style: bold claim, surprising stat, question, or story opener
- Depth: surface-level takeaway vs technical detail
- Emoji use: none, subtle, or expressive
- Thread structure: linear narrative, numbered insights, or pros/cons

{context}

Rules:
- First tweet must be a strong hook that stops the scroll
- Each tweet must be self-contained and under 280 characters
- Final tweet should include a call-to-action or reflection
- Create 6-8 tweets total

Return ONLY a JSON object — no markdown, no extra text:
{{"tweets": [{{"index": 1, "content": "..."}}], "hashtags": ["..."], "audience": "...", "hook_type": "..."}}"""

_LINKEDIN_PROMPT = """You are writing a LinkedIn post to share insights from an academic research paper.

Paper title: {title}
Creator intent: {description}

Inferred from the intent above, apply these attributes as you write:
- Audience: who will read this post (practitioners, executives, students, general public)
- Tone register: formal, conversational, inspirational, or analytical
- Post structure: hook → insight → implication → CTA, or story → lesson → reflection
- Personal framing: share as a reader reacting, an expert contextualising, or a neutral reporter
- Line breaks: dense paragraphs or punchy short lines (LinkedIn rewards white space)
- Depth: focus on headline finding, methodology, or real-world application

{context}

Rules:
- Open with a hook line that earns the "see more" click
- LinkedIn posts perform best at 150-300 words — aim for that range
- End with a question or call-to-action to drive comments
- Do NOT use generic filler phrases like "Exciting new research shows…"

Return ONLY a JSON object — no markdown, no extra text:
{{"post": "...", "hashtags": ["..."], "hook": "...", "audience": "...", "word_count": 0}}"""

_CAROUSEL_PROMPT = """You are designing a LinkedIn carousel to make an academic paper accessible and shareable.

Paper title: {title}
Creator intent: {description}
Visual style: {color_scheme}

Inferred from the intent above, apply these attributes:
- Audience level: expert, practitioner, or general public
- Slide density: text-heavy explanation vs minimal punchy bullets
- Narrative arc: problem → solution → evidence → takeaway, or top-N insights list
- Visual emphasis: data/stats, diagrams hints, or quotes

{context}

Slide type rules — you MUST use exactly these types, in this order:
  1. First slide: type = "cover"  (title + subtitle hook)
  2-6. Middle slides: type = "finding" (key insight), "method" (how it works),
       "stat" (a number/result), or "quote" (a direct insight worth highlighting)
  Last slide: type = "cta"  (what the reader should do next)

Each slide must have:
  - type: one of cover | finding | method | stat | quote | cta
  - title: short headline (max 8 words)
  - body: 2-4 bullet points or 2 short sentences
  - visual_hint: one sentence describing what image/graphic would work here

Create 6-8 slides total. First must be cover, last must be cta.

Return ONLY a JSON object — no markdown, no extra text:
{{"slides": [{{"type": "cover", "title": "...", "body": "...", "visual_hint": "..."}}], "hashtags": ["..."]}}"""

_STUDY_OUTLINE_PROMPT = """You are analyzing the following research paper to create a structured learning plan.

Paper title: {title}

{context}

Analyze the paper and identify the key concepts a student needs to learn, in the most logical teaching order.
Tailor the sections specifically to this paper's content — do not use a generic structure.

Return ONLY a JSON object with this exact structure:
{{
  "summary": "A 2-3 sentence overview of what the student will learn from this paper.",
  "sections": [
    {{"index": 0, "title": "Section title", "description": "What this section covers and why it matters"}},
    {{"index": 1, "title": "Section title", "description": "What this section covers and why it matters"}}
  ]
}}

Create 4-7 sections. No markdown, no extra text."""

_STUDY_SECTION_PROMPT = """You are teaching a student about a research paper, one section at a time.

Paper title: {title}
Current section: {section_title}
Section focus: {section_description}

{context}

Write a clear, engaging, well-structured lesson for this section. Use the following guidelines:
- Start with a brief orientation sentence explaining what this section covers
- Break down complex ideas using analogies and plain language
- Use markdown formatting: ## for subsections, **bold** for key terms, bullet points for lists
- Include concrete examples where helpful
- End with a brief 1-2 sentence summary of the key takeaway

Write the lesson now:"""

_FLASHCARD_PROMPT = """Based on the research paper below, generate flashcards to help a student test their understanding.

Paper title: {title}

{context}

Create 6-8 flashcards covering the most important concepts, terms, methods, and findings from this paper.
Each card should have a clear question on the front and a concise but complete answer on the back.

Return ONLY a JSON object:
{{
  "cards": [
    {{"front": "Question or term to define", "back": "Answer or definition"}},
    {{"front": "...", "back": "..."}}
  ]
}}

No markdown, no extra text."""

_TECHNICAL_SECTION_PROMPTS = {
    "overview": """Analyze this research paper and write the Overview section of a technical analysis.

Paper title: {title}
{context}

Cover:
- What problem this paper solves and why it matters to engineers
- The core approach or method at a high level
- Key results or contributions

Use markdown formatting. Be precise and technically accurate. 2-3 paragraphs.""",

    "concepts": """Continue the technical analysis of this paper with the Core Concepts section.

Paper title: {title}
{prev_sections}
{context}

Cover:
- All key technical concepts, algorithms, and data structures used
- Definitions and explanations of domain-specific terminology
- Mathematical or algorithmic foundations where relevant

Use markdown with **bold** for key terms, bullet points for concept lists. Be thorough.""",

    "architecture": """Continue the technical analysis with the System Architecture section.

Paper title: {title}
{prev_sections}
{context}

Cover:
- The overall system or model architecture
- Components and how they interact
- Data flow through the system

Include a Mermaid diagram if the architecture can be meaningfully represented as one.
Use a fenced mermaid code block:
```mermaid
graph TD
    A[Component] --> B[Component]
```

Use markdown formatting. Be specific about architectural decisions.""",

    "implementation": """Continue the technical analysis with the Implementation Details section.

Paper title: {title}
{prev_sections}
{context}

Cover:
- Key algorithms with pseudocode or code sketches where helpful
- Data structures and their design rationale
- API or interface design considerations
- Training procedures, hyperparameters, or configuration details if applicable

Use fenced code blocks for pseudocode/code. Use markdown formatting.""",

    "scalability": """Complete the technical analysis with the Scalability & Trade-offs section.

Paper title: {title}
{prev_sections}
{context}

Cover:
- Computational complexity and resource requirements
- Known limitations and failure modes
- Engineering trade-offs
- Practical considerations for production deployment
- Open problems or areas for improvement

Use markdown formatting. Be honest about limitations.""",
}


class OpenAIProvider(LLMProvider):
    def __init__(self):
        client_kwargs: dict = {"api_key": cfg.LLM_API_KEY, "max_retries": 0}
        if cfg.LLM_BASE_URL:
            client_kwargs["base_url"] = cfg.LLM_BASE_URL
            logger.info("OpenAI client using base_url: %s", cfg.LLM_BASE_URL)
        self._client     = OpenAI(**client_kwargs)
        self._model_name = cfg.LLM_MODEL

    def _call_sync(self, messages: list[dict], max_tokens: int = 2048) -> str:
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def _with_retry(self, messages: list[dict], max_tokens: int = 2048) -> str:
        """
        Acquire a rate-limit token, then run the blocking call in executor
        with exponential backoff retry.

        Rate limiter runs BEFORE the first attempt and before each retry
        so bursts of calls (e.g. Technical analysis, 5 sections) are
        spaced correctly and don't cascade into provider-side 429s.
        """
        loop     = asyncio.get_running_loop()
        last_exc = None
        for attempt in range(1, cfg.MAX_RETRIES + 1):
            await llm_rate_limiter.acquire()
            try:
                return await loop.run_in_executor(
                    None, lambda: self._call_sync(messages, max_tokens)
                )
            except Exception as exc:
                last_exc = exc
                if attempt < cfg.MAX_RETRIES:
                    error_str = str(exc)
                    wait = 60 if "429" in error_str else 2 ** attempt
                    logger.warning(
                        "OpenAI attempt %d/%d failed: %s — retrying in %ds",
                        attempt, cfg.MAX_RETRIES, exc, wait,
                    )
                    await asyncio.sleep(wait)
        raise RuntimeError(
            f"OpenAI call failed after {cfg.MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    def _build_messages(
        self, system: str, user_prompt: str, history: list[dict] | None = None
    ) -> list[dict]:
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})
        return messages

    # ── Chat ──────────────────────────────────────────────────────────────────

    async def chat_response(
        self, context, title, authors, history, message, level, mode="standard"
    ) -> str:
        safe_context  = sanitize_context(context)
        system_prompt = self.get_system_prompt(mode)
        system        = system_prompt + "\n\n" + _CHAT_PROMPT.format(
            title=title,
            authors=", ".join(authors) if authors else "Unknown",
            context=safe_context,
            level=level,
        )
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})
        return await self._with_retry(messages)

    # ── Social ────────────────────────────────────────────────────────────────

    async def generate_twitter_thread(self, context, title, description) -> dict:
        safe_context = sanitize_context(context)
        prompt = _TWITTER_PROMPT.format(
            title=title, description=description, context=safe_context
        )
        raw = await self._with_retry(self._build_messages(self.SYSTEM_PROMPT, prompt))
        return parse_json_response(raw)

    async def generate_linkedin_post(self, context, title, description) -> dict:
        safe_context = sanitize_context(context)
        prompt = _LINKEDIN_PROMPT.format(
            title=title, description=description, context=safe_context
        )
        raw = await self._with_retry(self._build_messages(self.SYSTEM_PROMPT, prompt))
        return parse_json_response(raw)

    async def generate_carousel_content(
        self, context, title, description, color_scheme
    ) -> dict:
        safe_context = sanitize_context(context)
        prompt = _CAROUSEL_PROMPT.format(
            title=title, description=description,
            color_scheme=color_scheme, context=safe_context,
        )
        raw = await self._with_retry(self._build_messages(self.SYSTEM_PROMPT, prompt))
        return parse_json_response(raw)

    # ── Study ─────────────────────────────────────────────────────────────────

    async def generate_study_outline(self, context, title) -> dict:
        safe_context = sanitize_context(context)
        prompt = _STUDY_OUTLINE_PROMPT.format(title=title, context=safe_context)
        raw = await self._with_retry(
            self._build_messages(self.STUDY_SYSTEM_PROMPT, prompt), max_tokens=1024
        )
        return parse_json_response(raw)

    async def generate_study_section(
        self, context, title, section_title, section_description
    ) -> str:
        safe_context = sanitize_context(context)
        prompt = _STUDY_SECTION_PROMPT.format(
            title=title, section_title=section_title,
            section_description=section_description, context=safe_context,
        )
        return await self._with_retry(
            self._build_messages(self.STUDY_SYSTEM_PROMPT, prompt), max_tokens=2048
        )

    async def generate_flashcards(self, context, title) -> dict:
        safe_context = sanitize_context(context)
        prompt = _FLASHCARD_PROMPT.format(title=title, context=safe_context)
        raw = await self._with_retry(
            self._build_messages(self.STUDY_SYSTEM_PROMPT, prompt), max_tokens=1500
        )
        return parse_json_response(raw)

    # ── Technical ─────────────────────────────────────────────────────────────

    async def generate_technical_section(
        self, context, title, section_key, section_label, prev_sections
    ) -> str:
        safe_context = sanitize_context(context)
        template = _TECHNICAL_SECTION_PROMPTS[section_key]
        prompt = template.format(
            title=title, context=safe_context,
            prev_sections=prev_sections if prev_sections else "",
        )
        return await self._with_retry(
            self._build_messages(self.TECHNICAL_SYSTEM_PROMPT, prompt), max_tokens=2048
        )