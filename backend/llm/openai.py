"""
openai.py — OpenAI implementation of LLMProvider.

Also used for OpenRouter — pass LLM_BASE_URL=https://openrouter.ai/api/v1
and LLM_PROVIDER=openrouter to route through OpenRouter's unified API.

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
        client_kwargs: dict = {
            "api_key":     cfg.LLM_API_KEY,
            # Disable the SDK's built-in retry entirely.
            # When using OpenRouter free tier, the SDK's internal retry fires
            # multiple times before our own retry loop runs, multiplying total
            # attempts and burning through the rate limit faster.
            # Our _with_retry() already handles retries correctly.
            "max_retries": 0,
        }
        if cfg.LLM_BASE_URL:
            client_kwargs["base_url"] = cfg.LLM_BASE_URL
            logger.info("OpenAI client using base_url: %s", cfg.LLM_BASE_URL)

        self._client     = OpenAI(**client_kwargs)
        self._model_name = cfg.LLM_MODEL

    def _call_sync(self, messages: list[dict]) -> str:
        """Blocking OpenAI chat completion call."""
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    async def _with_retry(self, messages: list[dict]) -> str:
        """
        Run the blocking call in executor with exponential backoff retry.

        On 429 from OpenRouter free tier, waits 60 seconds between attempts
        to respect the upstream rate limit window (typically 60 seconds).
        On other transient errors, uses standard exponential backoff.
        """
        loop = asyncio.get_running_loop()
        last_exc = None
        for attempt in range(1, cfg.MAX_RETRIES + 1):
            try:
                return await loop.run_in_executor(None, self._call_sync, messages)
            except Exception as exc:
                last_exc = exc
                if attempt < cfg.MAX_RETRIES:
                    error_str = str(exc)
                    # OpenRouter free tier rate limit windows are ~60 seconds.
                    # A 15s wait was not enough — all 3 retries hit the same
                    # rate limit window and failed. 60s clears it reliably.
                    wait = 60 if "429" in error_str else 2 ** attempt
                    logger.warning(
                        "OpenAI attempt %d/%d failed: %s — retrying in %ds",
                        attempt, cfg.MAX_RETRIES, exc, wait,
                    )
                    await asyncio.sleep(wait)
        raise RuntimeError(
            f"OpenAI call failed after {cfg.MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

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
        raw = await self._with_retry(self._build_messages(prompt))
        return parse_json_response(raw)

    async def generate_linkedin_post(self, context, title, style, tone) -> dict:
        safe_context = sanitize_context(context)
        prompt = _LINKEDIN_PROMPT.format(
            title=title, style=style, tone=tone, context=safe_context
        )
        raw = await self._with_retry(self._build_messages(prompt))
        return parse_json_response(raw)

    async def generate_carousel_content(self, context, title, style, tone, color_scheme) -> dict:
        safe_context = sanitize_context(context)
        prompt = _CAROUSEL_PROMPT.format(
            title=title, style=style, tone=tone,
            color_scheme=color_scheme, context=safe_context
        )
        raw = await self._with_retry(self._build_messages(prompt))
        return parse_json_response(raw)