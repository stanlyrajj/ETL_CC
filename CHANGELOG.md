# Changelog

## [1.0.0] — 2026-03-16

### Added

- **Search and fetch** — Fetch papers from arXiv (via the official Python SDK) and PubMed (via NCBI E-utilities) by topic with optional date filtering and configurable result limits
- **Local PDF upload** — Upload any PDF up to 50 MB directly from the browser
- **Full processing pipeline** — Automatic extraction (Docling), chunking, embedding (all-MiniLM-L6-v2), and vector storage (ChromaDB) for every paper
- **Real-time progress** — Server-Sent Events stream live pipeline stage updates to the frontend; stuck papers are automatically re-queued on server restart
- **RAG chat** — Ask questions about any processed paper with retrieval-augmented responses; configurable teaching level (beginner / intermediate / advanced)
- **Twitter thread generation** — Generate 5–8 tweet threads with hashtags; tweets automatically truncated to 280 characters
- **LinkedIn post generation** — Generate professional posts with hook and hashtags
- **LinkedIn carousel generation** — Generate 6–8 slide carousels with cover and CTA enforcement; export as 1080×1080pt PDF in dark, light, or bold color schemes
- **One-click sharing** — Deep-links to LinkedIn and Twitter/X composers with content pre-filled
- **Configurable LLM** — Switch between Google Gemini, OpenAI, and Anthropic by setting one environment variable; retry logic with exponential backoff
- **Security hardening** — Prompt-injection stripping at ingestion and LLM boundaries; file type and magic-byte validation; system prompt prohibits credential disclosure
- **Backend validation suite** — 255-check automated test suite covering config, database, security, pipeline, chat, content generation, export, API routes, and HTTP endpoints
