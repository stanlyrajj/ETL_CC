"""
factory.py — Returns the correct LLMProvider instance based on LLM_PROVIDER config.

Lazy initialization: the provider is created on first call, not at import time.
set_model() allows switching the active model at runtime without restarting.

Supported providers:
  openai      — OpenAI API directly
  anthropic   — Anthropic API directly
  gemini      — Google Gemini API directly
  groq        — Groq API (OpenAI-compatible, set LLM_BASE_URL=https://api.groq.com/openai/v1)
  openrouter  — OpenRouter API (OpenAI-compatible, set LLM_BASE_URL=https://openrouter.ai/api/v1)
"""

import logging

from config import cfg
from llm.base import LLMProvider

logger = logging.getLogger(__name__)

_SUPPORTED_PROVIDERS = ("openai", "gemini", "anthropic", "groq", "openrouter")

# Module-level cache — created once on first call to get_provider()
_instance: LLMProvider | None = None


def get_provider() -> LLMProvider:
    """
    Return the shared LLMProvider instance for the configured provider.

    Raises ValueError with a clear message if LLM_PROVIDER is not recognized.
    """
    global _instance

    if _instance is not None:
        return _instance

    provider = cfg.LLM_PROVIDER.lower()

    if provider == "openai":
        from llm.openai import OpenAIProvider
        _instance = OpenAIProvider()

    elif provider == "gemini":
        from llm.gemini import GeminiProvider
        _instance = GeminiProvider()

    elif provider == "anthropic":
        from llm.anthropic import AnthropicProvider
        _instance = AnthropicProvider()

    elif provider in ("groq", "openrouter"):
        # Both Groq and OpenRouter are OpenAI-compatible — reuse OpenAIProvider
        # with the appropriate base_url set in config.
        # Groq:       LLM_BASE_URL=https://api.groq.com/openai/v1
        # OpenRouter: LLM_BASE_URL=https://openrouter.ai/api/v1
        from llm.openai import OpenAIProvider
        _instance = OpenAIProvider()

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            f"Must be one of: {', '.join(_SUPPORTED_PROVIDERS)}. "
            "Set the correct value in your .env file."
        )

    logger.info(
        "LLM provider initialized: %s (model=%s%s)",
        provider,
        cfg.LLM_MODEL,
        f" via {cfg.LLM_BASE_URL}" if cfg.LLM_BASE_URL else "",
    )
    return _instance


def set_model(model_id: str) -> None:
    """
    Switch the active model and reset the provider cache.

    The next call to get_provider() will create a fresh provider instance
    using the new model name. Used by POST /api/models/select.
    """
    global _instance
    cfg.LLM_MODEL = model_id
    _instance = None
    logger.info("LLM model switched to: %s", model_id)
