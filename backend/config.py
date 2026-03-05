"""
config.py — Central configuration with startup validation.

Fixes applied:
  ARCH-03: All os.getenv() calls moved into __init__ — tests can patch os.environ before
           instantiation and see the change in Config().FIELD.
  SEC-05:  API_SECRET_KEY field added — enables optional API key auth in dev/prod.
  PERF-01: MAX_EMBEDDING_BATCH_SIZE field added — caps per-batch embedding size.
"""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Config:
    def __init__(self):
        """Read all env vars at instantiation time — enables os.environ patching in tests."""

        # ── Database ──────────────────────────────────────────────────────────
        self.DATABASE_URL: str = os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/researchrag"
        )

        # ── Gemini (Google AI Studio — free tier) ─────────────────────────────
        self.GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
        self.GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

        # SEC-05: optional API authentication key
        # Set to a long random string in production. Empty string = dev mode (no auth).
        self.API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "")

        # ── Vector store ──────────────────────────────────────────────────────
        self.CHROMA_DIR: str = str(
            Path(os.getenv("CHROMA_DIR", "./chroma_db")).resolve()
        )

        # ── File storage ──────────────────────────────────────────────────────
        self.DOWNLOADS_DIR: str = str(
            Path(os.getenv("DOWNLOADS_DIR", "./downloads")).resolve()
        )
        self.CAROUSEL_OUTPUT_DIR: str = str(
            Path(os.getenv("CAROUSEL_OUTPUT_DIR", "./carousel_outputs")).resolve()
        )

        # ── Rate limits ───────────────────────────────────────────────────────
        self.ARXIV_RATE_LIMIT_SECONDS: float = float(os.getenv("ARXIV_RATE_LIMIT", "3.0"))
        self.PUBMED_RATE_LIMIT_SECONDS: float = float(os.getenv("PUBMED_RATE_LIMIT", "0.4"))

        # ── RAG chunking ─────────────────────────────────────────────────────
        self.CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.MIN_CHUNK_CHARS: int = int(os.getenv("MIN_CHUNK_CHARS", "50"))

        # PERF-01: cap per-batch embedding size to prevent OOM on large papers
        self.MAX_EMBEDDING_BATCH_SIZE: int = int(os.getenv("MAX_EMBEDDING_BATCH_SIZE", "50"))

        # ── Embedding model (local, free) ─────────────────────────────────────
        self.EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

        # ── API server ────────────────────────────────────────────────────────
        self.API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.CORS_ORIGINS: list = os.getenv(
            "CORS_ORIGINS", "http://localhost:3000"
        ).split(",")

        # ── Retry policy ──────────────────────────────────────────────────────
        self.MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
        self.RETRY_BACKOFF_BASE: float = float(os.getenv("RETRY_BACKOFF_BASE", "2.0"))

    def validate(self) -> None:
        errors = []
        warnings = []

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

        for d in [self.DOWNLOADS_DIR, self.CHROMA_DIR, self.CAROUSEL_OUTPUT_DIR]:
            Path(d).mkdir(parents=True, exist_ok=True)

        for w in warnings:
            logger.warning(f"[Config] ⚠  {w}")

        if errors:
            for e in errors:
                logger.error(f"[Config] ✗  {e}")
            sys.exit(1)

        logger.info("[Config] ✓  All required configuration present.")
        if self.API_SECRET_KEY:
            logger.info("[Config] ✓  API authentication enabled (X-API-Key required).")
        else:
            logger.warning("[Config] ⚠  API_SECRET_KEY not set — running in dev mode (no auth).")

    def _pg_reachable(self) -> bool:
        import socket
        try:
            host = "localhost"
            port = 5432
            if "@" in self.DATABASE_URL:
                netloc = self.DATABASE_URL.split("@")[-1].split("/")[0]
                if ":" in netloc:
                    host, port_str = netloc.rsplit(":", 1)
                    port = int(port_str)
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            return False


cfg = Config()