"""
factory.py — Returns the correct LLMProvider instance based on LLM_PROVIDER config.

Lazy initialization: the provider is created on first call, not at import time.
"""

import logging

from config import cfg
from llm.base import LLMProvider

logger = logging.getLogger(__name__)

_SUPPORTED_PROVIDERS = ("openai", "gemini", "anthropic")

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

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            f"Must be one of: {', '.join(_SUPPORTED_PROVIDERS)}. "
            "Set the correct value in your .env file."
        )

    logger.info("LLM provider initialized: %s (model=%s)", provider, cfg.LLM_MODEL)
    return _instance
