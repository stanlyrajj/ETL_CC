"""
gemini.py — Google Gemini implementation of LLMProvider.

Uses google-genai (the current supported SDK, replacing google-generativeai).
Install: pip install google-genai

All SDK calls run in run_in_executor since the sync client is blocking.
All paper context is sanitized before use in prompts.
Retries on transient errors up to MAX_RETRIES.
"""

import asyncio
import logging

from google import genai
from google.genai import types

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
- advanced: use technical language freely

Respond to the user's message based only on the paper content above."""

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


class GeminiProvider(LLMProvider):
    def __init__(self):
        self._client     = genai.Client(api_key=cfg.LLM_API_KEY)
        self._model_name = cfg.LLM_MODEL

    def _chat_sync(self, system: str, history: list[dict], message: str) -> str:
        """Blocking Gemini chat call using the google-genai SDK."""
        # Convert history to google-genai Content objects
        contents = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

        # Append the new user message
        contents.append(types.Content(role="user", parts=[types.Part(text=message)]))

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        return response.text or ""

    def _generate_sync(self, system: str, prompt: str) -> str:
        """Blocking Gemini single-turn generation call."""
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        return response.text or ""

    async def _with_retry(self, fn, *args):
        """Run a blocking function in executor with retry on transient errors."""
        # FIX L1: Use get_running_loop() — get_event_loop() is deprecated in
        # Python 3.10+ and may raise RuntimeError inside coroutines in Python 3.12.
        loop = asyncio.get_running_loop()
        last_exc = None
        for attempt in range(1, cfg.MAX_RETRIES + 1):
            try:
                return await loop.run_in_executor(None, fn, *args)
            except Exception as exc:
                last_exc = exc
                if attempt < cfg.MAX_RETRIES:
                    wait = 2 ** attempt
                    logger.warning(
                        "Gemini attempt %d/%d failed: %s — retrying in %ds",
                        attempt, cfg.MAX_RETRIES, exc, wait,
                    )
                    await asyncio.sleep(wait)
        raise RuntimeError(
            f"Gemini call failed after {cfg.MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    async def chat_response(self, context, title, authors, history, message, level) -> str:
        safe_context = sanitize_context(context)
        system = self.SYSTEM_PROMPT + "\n\n" + _CHAT_PROMPT.format(
            title=title,
            authors=", ".join(authors) if authors else "Unknown",
            context=safe_context,
            level=level,
        )
        return await self._with_retry(self._chat_sync, system, history, message)

    async def generate_twitter_thread(self, context, title, style, tone) -> dict:
        safe_context = sanitize_context(context)
        prompt = _TWITTER_PROMPT.format(
            title=title, style=style, tone=tone, context=safe_context
        )
        raw = await self._with_retry(self._generate_sync, self.SYSTEM_PROMPT, prompt)
        return parse_json_response(raw)

    async def generate_linkedin_post(self, context, title, style, tone) -> dict:
        safe_context = sanitize_context(context)
        prompt = _LINKEDIN_PROMPT.format(
            title=title, style=style, tone=tone, context=safe_context
        )
        raw = await self._with_retry(self._generate_sync, self.SYSTEM_PROMPT, prompt)
        return parse_json_response(raw)

    async def generate_carousel_content(self, context, title, style, tone, color_scheme) -> dict:
        safe_context = sanitize_context(context)
        prompt = _CAROUSEL_PROMPT.format(
            title=title, style=style, tone=tone,
            color_scheme=color_scheme, context=safe_context
        )
        raw = await self._with_retry(self._generate_sync, self.SYSTEM_PROMPT, prompt)
        return parse_json_response(raw)