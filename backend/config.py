"""
config.py — Central configuration with startup validation.
Import this everywhere instead of reading os.getenv() directly.

Usage:
    from config import cfg
    print(cfg.DATABASE_URL)
    print(cfg.GEMINI_API_KEY)
"""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Config:
    # ── Database ──────────────────────────────────────────────────────────
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/researchrag"
    )

    # ── Gemini (Google AI Studio — free tier) ─────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")   # free tier model

    # ── Vector store ──────────────────────────────────────────────────────
    CHROMA_DIR: str = str(
        Path(os.getenv("CHROMA_DIR", "./chroma_db")).resolve()
    )

    # ── File storage ──────────────────────────────────────────────────────
    DOWNLOADS_DIR: str = str(
        Path(os.getenv("DOWNLOADS_DIR", "./downloads")).resolve()
    )
    CAROUSEL_OUTPUT_DIR: str = str(
        Path(os.getenv("CAROUSEL_OUTPUT_DIR", "./carousel_outputs")).resolve()
    )

    # ── Rate limits ───────────────────────────────────────────────────────
    ARXIV_RATE_LIMIT_SECONDS: float = float(os.getenv("ARXIV_RATE_LIMIT", "3.0"))
    PUBMED_RATE_LIMIT_SECONDS: float = float(os.getenv("PUBMED_RATE_LIMIT", "0.4"))

    # ── RAG chunking ─────────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    MIN_CHUNK_CHARS: int = int(os.getenv("MIN_CHUNK_CHARS", "50"))

    # ── Embedding model (local, free) ─────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ── API server ────────────────────────────────────────────────────────
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
    CORS_ORIGINS: list = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000"
    ).split(",")

    # ── Retry policy ──────────────────────────────────────────────────────
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_BACKOFF_BASE: float = float(os.getenv("RETRY_BACKOFF_BASE", "2.0"))

    def validate(self) -> None:
        """
        Called once at application startup.
        Logs warnings for missing optional vars.
        Exits for missing critical vars.
        """
        errors = []
        warnings = []

        # Critical
        if not self.GEMINI_API_KEY:
            errors.append(
                "GEMINI_API_KEY is not set. "
                "Get a free key at https://aistudio.google.com/app/apikey"
            )

        if "localhost" in self.DATABASE_URL and not self._pg_reachable():
            warnings.append(
                "PostgreSQL does not appear to be running at the configured DATABASE_URL. "
                "The app will start but DB operations will fail."
            )

        # Ensure directories exist
        for d in [self.DOWNLOADS_DIR, self.CHROMA_DIR, self.CAROUSEL_OUTPUT_DIR]:
            Path(d).mkdir(parents=True, exist_ok=True)

        for w in warnings:
            logger.warning(f"[Config] ⚠  {w}")

        if errors:
            for e in errors:
                logger.error(f"[Config] ✗  {e}")
            sys.exit(1)

        logger.info("[Config] ✓  All required configuration present.")

    def _pg_reachable(self) -> bool:
        """Quick TCP check for PostgreSQL reachability (sync, used only at startup)."""
        import socket
        try:
            host = "localhost"
            port = 5432
            # Parse port from URL if custom
            if "@" in self.DATABASE_URL:
                netloc = self.DATABASE_URL.split("@")[-1].split("/")[0]
                if ":" in netloc:
                    host, port_str = netloc.rsplit(":", 1)
                    port = int(port_str)
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            return False


# Module-level singleton
cfg = Config()