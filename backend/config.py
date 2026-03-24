import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load backend/.env before reading any env vars
load_dotenv(Path(__file__).parent / ".env")

# Default models per provider
_DEFAULT_MODELS = {
    "openai":     "gpt-4o-mini",
    "gemini":     "gemini-2.0-flash",
    "anthropic":  "claude-sonnet-4-20250514",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
}

# Valid providers — checked in validate()
_VALID_PROVIDERS = {"openai", "gemini", "anthropic", "openrouter"}

# Curated list of free OpenRouter models tested with this codebase.
# Exposed via GET /api/models when LLM_PROVIDER=openrouter.
AVAILABLE_MODELS = [
    {
        "id":          "meta-llama/llama-3.3-70b-instruct:free",
        "name":        "Llama 3.3 70B",
        "provider":    "Meta",
        "description": "Best quality. Strong reasoning and JSON output.",
        "recommended": True,
    },
    {
        "id":          "google/gemini-2.0-flash-exp:free",
        "name":        "Gemini 2.0 Flash (free)",
        "provider":    "Google",
        "description": "Fast and capable. Good for chat and generation.",
        "recommended": False,
    },
    {
        "id":          "mistralai/mistral-7b-instruct:free",
        "name":        "Mistral 7B",
        "provider":    "Mistral",
        "description": "Lightweight and fast. May occasionally miss JSON structure.",
        "recommended": False,
    },
    {
        "id":          "qwen/qwen-2.5-72b-instruct:free",
        "name":        "Qwen 2.5 72B",
        "provider":    "Alibaba",
        "description": "Strong alternative to Llama. Good multilingual support.",
        "recommended": False,
    },
]


def _int_env(key: str, default: str) -> int:
    """Cast an env var to int; exit with a clear message on bad value."""
    raw = os.getenv(key, default)
    try:
        return int(raw)
    except ValueError:
        print(f"Configuration error: {key}={raw!r} is not a valid integer.", file=sys.stderr)
        sys.exit(1)


def _float_env(key: str, default: str) -> float:
    """Cast an env var to float; exit with a clear message on bad value."""
    raw = os.getenv(key, default)
    try:
        return float(raw)
    except ValueError:
        print(f"Configuration error: {key}={raw!r} is not a valid float.", file=sys.stderr)
        sys.exit(1)


class Config:
    def __init__(self):
        # ── Required ──────────────────────────────────────────────────────────
        self.LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "")
        self.LLM_API_KEY:  str = os.getenv("LLM_API_KEY", "")

        # ── LLM model (default depends on provider) ───────────────────────────
        default_model = _DEFAULT_MODELS.get(self.LLM_PROVIDER.lower(), "")
        self.LLM_MODEL: str = os.getenv("LLM_MODEL", default_model)

        # ── LLM base URL ──────────────────────────────────────────────────────
        # Leave unset for direct provider access (OpenAI, Gemini, Anthropic).
        # Required when LLM_PROVIDER=openrouter: set to https://openrouter.ai/api/v1
        self.LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")

        # ── Database ──────────────────────────────────────────────────────────
        self.DATABASE_URL: str = os.getenv(
            "DATABASE_URL", "sqlite+aiosqlite:///./researchrag.db"
        )

        # ── Directories ───────────────────────────────────────────────────────
        self.CHROMA_DIR:          str = os.getenv("CHROMA_DIR",          "./chroma_db")
        self.DOWNLOADS_DIR:       str = os.getenv("DOWNLOADS_DIR",       "./downloads")
        self.CAROUSEL_OUTPUT_DIR: str = os.getenv("CAROUSEL_OUTPUT_DIR", "./carousel_outputs")

        # ── Rate limits ───────────────────────────────────────────────────────
        self.ARXIV_RATE_LIMIT:  float = _float_env("ARXIV_RATE_LIMIT",  "3.0")
        self.PUBMED_RATE_LIMIT: float = _float_env("PUBMED_RATE_LIMIT", "0.4")

        # ── Chunking ──────────────────────────────────────────────────────────
        self.CHUNK_SIZE:      int = _int_env("CHUNK_SIZE",      "1000")
        self.CHUNK_OVERLAP:   int = _int_env("CHUNK_OVERLAP",   "200")
        self.MIN_CHUNK_CHARS: int = _int_env("MIN_CHUNK_CHARS", "50")

        # ── Embedding ─────────────────────────────────────────────────────────
        self.EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

        # ── Extraction ────────────────────────────────────────────────────────
        self.DOCLING_FAST_TIMEOUT: int = _int_env("DOCLING_FAST_TIMEOUT", "60")

        # ── CORS ──────────────────────────────────────────────────────────────
        self.CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://localhost:3000")

        # ── Retries ───────────────────────────────────────────────────────────
        self.MAX_RETRIES: int = _int_env("MAX_RETRIES", "3")

    def validate(self):
        """Exit with a clear error if required variables are missing or invalid."""
        errors = []

        if not self.LLM_PROVIDER:
            errors.append(
                "LLM_PROVIDER is not set (required). "
                f"Must be one of: {', '.join(sorted(_VALID_PROVIDERS))}."
            )
        elif self.LLM_PROVIDER.lower() not in _VALID_PROVIDERS:
            errors.append(
                f"LLM_PROVIDER={self.LLM_PROVIDER!r} is not supported. "
                f"Must be one of: {', '.join(sorted(_VALID_PROVIDERS))}."
            )

        if not self.LLM_API_KEY:
            errors.append("LLM_API_KEY is not set (required).")

        if not self.LLM_MODEL:
            errors.append(
                f"LLM_MODEL could not be determined for provider "
                f"{self.LLM_PROVIDER!r}. Set LLM_MODEL explicitly in your .env file."
            )

        if self.LLM_PROVIDER.lower() == "openrouter" and not self.LLM_BASE_URL:
            errors.append(
                "LLM_BASE_URL is required when LLM_PROVIDER=openrouter. "
                "Set it to: https://openrouter.ai/api/v1"
            )

        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            errors.append(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be less than "
                f"CHUNK_SIZE ({self.CHUNK_SIZE})."
            )

        if errors:
            print("Configuration error(s):", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            sys.exit(1)

    def create_dirs(self):
        """Create output directories if they don't exist."""
        for path in (self.CHROMA_DIR, self.DOWNLOADS_DIR, self.CAROUSEL_OUTPUT_DIR):
            Path(path).mkdir(parents=True, exist_ok=True)


# Single shared instance — import this everywhere
cfg = Config()