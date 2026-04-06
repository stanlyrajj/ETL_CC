"""
gemini.py — Google Gemini implementation of LLMProvider.
"""

import asyncio
import logging

from google import genai
from google.genai import types

from config import cfg
from llm.base import LLMProvider, parse_json_response, sanitize_context
from llm.openai import (
    _CHAT_PROMPT, _TWITTER_PROMPT, _LINKEDIN_PROMPT, _CAROUSEL_PROMPT,
    _STUDY_OUTLINE_PROMPT, _STUDY_SECTION_PROMPT, _FLASHCARD_PROMPT,
    _TECHNICAL_SECTION_PROMPTS,
)

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    def __init__(self):
        self._client     = genai.Client(api_key=cfg.LLM_API_KEY)
        self._model_name = cfg.LLM_MODEL

    def _chat_sync(self, system: str, history: list[dict], message: str, max_tokens: int = 2048) -> str:
        contents = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
        contents.append(types.Content(role="user", parts=[types.Part(text=message)]))
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""

    def _generate_sync(self, system: str, prompt: str, max_tokens: int = 2048) -> str:
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""

    async def _with_retry(self, fn, *args):
        loop     = asyncio.get_running_loop()
        last_exc = None
        for attempt in range(1, cfg.MAX_RETRIES + 1):
            try:
                return await loop.run_in_executor(None, fn, *args)
            except Exception as exc:
                last_exc = exc
                if attempt < cfg.MAX_RETRIES:
                    wait = 2 ** attempt
                    logger.warning("Gemini attempt %d/%d failed: %s — retrying in %ds",
                                   attempt, cfg.MAX_RETRIES, exc, wait)
                    await asyncio.sleep(wait)
        raise RuntimeError(f"Gemini call failed after {cfg.MAX_RETRIES} attempts: {last_exc}") from last_exc

    # ── Chat ──────────────────────────────────────────────────────────────────

    async def chat_response(self, context, title, authors, history, message, level, mode="standard") -> str:
        safe_context  = sanitize_context(context)
        system_prompt = self.get_system_prompt(mode)
        system        = system_prompt + "\n\n" + _CHAT_PROMPT.format(
            title=title, authors=", ".join(authors) if authors else "Unknown",
            context=safe_context, level=level,
        )
        return await self._with_retry(self._chat_sync, system, history, message)

    # ── Social ────────────────────────────────────────────────────────────────

    async def generate_twitter_thread(self, context, title, style, tone) -> dict:
        safe_context = sanitize_context(context)
        prompt = _TWITTER_PROMPT.format(title=title, style=style, tone=tone, context=safe_context)
        raw = await self._with_retry(self._generate_sync, self.SYSTEM_PROMPT, prompt)
        return parse_json_response(raw)

    async def generate_linkedin_post(self, context, title, style, tone) -> dict:
        safe_context = sanitize_context(context)
        prompt = _LINKEDIN_PROMPT.format(title=title, style=style, tone=tone, context=safe_context)
        raw = await self._with_retry(self._generate_sync, self.SYSTEM_PROMPT, prompt)
        return parse_json_response(raw)

    async def generate_carousel_content(self, context, title, style, tone, color_scheme) -> dict:
        safe_context = sanitize_context(context)
        prompt = _CAROUSEL_PROMPT.format(title=title, style=style, tone=tone,
                                         color_scheme=color_scheme, context=safe_context)
        raw = await self._with_retry(self._generate_sync, self.SYSTEM_PROMPT, prompt)
        return parse_json_response(raw)

    # ── Study ─────────────────────────────────────────────────────────────────

    async def generate_study_outline(self, context, title) -> dict:
        safe_context = sanitize_context(context)
        prompt = _STUDY_OUTLINE_PROMPT.format(title=title, context=safe_context)
        raw = await self._with_retry(self._generate_sync, self.STUDY_SYSTEM_PROMPT, prompt, 1024)
        return parse_json_response(raw)

    async def generate_study_section(self, context, title, section_title, section_description) -> str:
        safe_context = sanitize_context(context)
        prompt = _STUDY_SECTION_PROMPT.format(
            title=title, section_title=section_title,
            section_description=section_description, context=safe_context,
        )
        return await self._with_retry(self._generate_sync, self.STUDY_SYSTEM_PROMPT, prompt, 2048)

    async def generate_flashcards(self, context, title) -> dict:
        safe_context = sanitize_context(context)
        prompt = _FLASHCARD_PROMPT.format(title=title, context=safe_context)
        raw = await self._with_retry(self._generate_sync, self.STUDY_SYSTEM_PROMPT, prompt, 1500)
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
        return await self._with_retry(self._generate_sync, self.TECHNICAL_SYSTEM_PROMPT, prompt, 2048)
