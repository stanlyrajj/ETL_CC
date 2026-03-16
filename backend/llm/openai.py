"""
openai.py — OpenAI implementation of LLMProvider.

All SDK calls run in run_in_executor (blocking SDK).
All paper context is sanitized before use in prompts.
Retries on transient errors up to MAX_RETRIES.
"""

import asyncio
import logging

from openai import OpenAI

from config import cfg
from llm.base import LLMProvider, parse_json_response, sanitize_context

logger = logging.getLogger(__name__)

# Prompt templates ─────────────────────────────────────────────────────────────

_CHAT_PROMPT = """You are helping a user understand the following research paper.

Paper title: {title}
Authors: {authors}

{context}

The user's knowledge level is: {level}
- beginner: use plain language, avoid jargon, use analogies
- intermediate: assume basic domain knowledge
- advanced: use technical language freely"""

_TWITTER_PROMPT = """Based on the research paper below, create an engaging Twitter/X thread.

Paper title: {title}
Style: {style}
Tone: {tone}

{context}

Return ONLY a JSON object with this exact structure:
{{
  "tweets": [
    {{"index": 1, "content": "tweet text here"}},
    {{"index": 2, "content": "tweet text here"}}
  ],
  "hashtags": ["hashtag1", "hashtag2"]
}}

Each tweet must be under 280 characters. Create 5-8 tweets. No markdown, no extra text."""

_LINKEDIN_PROMPT = """Based on the research paper below, write a professional LinkedIn post.

Paper title: {title}
Style: {style}
Tone: {tone}

{context}

Return ONLY a JSON object with this exact structure:
{{
  "post": "full post text here",
  "hashtags": ["hashtag1", "hashtag2"],
  "hook": "opening line that grabs attention"
}}

No markdown, no extra text."""

_CAROUSEL_PROMPT = """Based on the research paper below, create LinkedIn carousel slide content.

Paper title: {title}
Style: {style}
Tone: {tone}
Color scheme: {color_scheme}

{context}

Return ONLY a JSON object with this exact structure:
{{
  "slides": [
    {{
      "index": 1,
      "heading": "slide heading",
      "body": "slide body text",
      "speaker_note": "what to say about this slide"
    }}
  ],
  "title": "carousel title",
  "subtitle": "carousel subtitle"
}}

Create 6-8 slides. Keep each slide concise. No markdown, no extra text."""


class OpenAIProvider(LLMProvider):
    def __init__(self):
        self._client     = OpenAI(api_key=cfg.LLM_API_KEY)
        self._model_name = cfg.LLM_MODEL

    def _call_sync(self, messages: list[dict]) -> str:
        """Blocking OpenAI chat completion call."""
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    async def _with_retry(self, messages: list[dict]) -> str:
        """Run the blocking call in executor with retry on transient errors."""
        # FIX L1: Use get_running_loop() — get_event_loop() is deprecated in
        # Python 3.10+ and may raise RuntimeError inside coroutines in Python 3.12.
        loop = asyncio.get_running_loop()
        last_exc = None
        for attempt in range(1, cfg.MAX_RETRIES + 1):
            try:
                return await loop.run_in_executor(None, self._call_sync, messages)
            except Exception as exc:
                last_exc = exc
                if attempt < cfg.MAX_RETRIES:
                    wait = 2 ** attempt
                    logger.warning("OpenAI attempt %d/%d failed: %s — retrying in %ds",
                                   attempt, cfg.MAX_RETRIES, exc, wait)
                    await asyncio.sleep(wait)
        raise RuntimeError(f"OpenAI call failed after {cfg.MAX_RETRIES} attempts: {last_exc}") from last_exc

    def _build_messages(self, user_prompt: str, history: list[dict] | None = None) -> list[dict]:
        """Build the messages array with system prompt, history, and user message."""
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})
        return messages

    async def chat_response(self, context, title, authors, history, message, level) -> str:
        safe_context = sanitize_context(context)
        system_context = _CHAT_PROMPT.format(
            title=title,
            authors=", ".join(authors) if authors else "Unknown",
            context=safe_context,
            level=level,
        )
        # Prepend paper context to system prompt
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT + "\n\n" + system_context}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})
        return await self._with_retry(messages)

    async def generate_twitter_thread(self, context, title, style, tone) -> dict:
        safe_context = sanitize_context(context)
        prompt = _TWITTER_PROMPT.format(
            title=title, style=style, tone=tone, context=safe_context
        )
        messages = self._build_messages(prompt)
        raw = await self._with_retry(messages)
        return parse_json_response(raw)

    async def generate_linkedin_post(self, context, title, style, tone) -> dict:
        safe_context = sanitize_context(context)
        prompt = _LINKEDIN_PROMPT.format(
            title=title, style=style, tone=tone, context=safe_context
        )
        messages = self._build_messages(prompt)
        raw = await self._with_retry(messages)
        return parse_json_response(raw)

    async def generate_carousel_content(self, context, title, style, tone, color_scheme) -> dict:
        safe_context = sanitize_context(context)
        prompt = _CAROUSEL_PROMPT.format(
            title=title, style=style, tone=tone,
            color_scheme=color_scheme, context=safe_context
        )
        messages = self._build_messages(prompt)
        raw = await self._with_retry(messages)
        return parse_json_response(raw)