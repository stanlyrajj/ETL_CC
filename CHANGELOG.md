# Changelog

All notable changes to ResearchRAG are documented here.

---

## [1.1.0] — 2026-05-06

### Added

- **Study mode persistence** — Generated learning plans, section explanations, and flashcards are now cached in SQLite and restored instantly on revisit. No re-generation on page reload.
- **Study mode navigation (Option D)** — Outline screen is now a full table of contents with clickable sections. Completed sections show a checkmark and can be reviewed at any time. A sticky header on the teaching screen shows a "☰ Outline" back button and dot navigator for jumping between sections directly.
- **Technical mode persistence** — Technical analysis sections cached per paper in SQLite. Returns instantly on revisit. Regenerate button clears the cache explicitly.
- **Resizable panels** — The content generation panel (right side) can be dragged to any width between 240–680px. The chat input bar can be dragged taller for longer messages, switching to a `<textarea>` with Shift+Enter support above a height threshold.
- **Chat history persistence** — Chat sessions now sync from the database after every send, eliminating ephemeral message IDs and ensuring message history survives page reloads exactly as stored.
- **Token authentication** — Optional `APP_TOKEN` environment variable. When set, all API endpoints require an `X-App-Token` header. The frontend reads `NEXT_PUBLIC_APP_TOKEN` and sends it automatically.
- **Loopback-only binding** — Server now binds to `127.0.0.1` by default. Set `HOST=0.0.0.0` explicitly to expose over a network (a warning is printed if `APP_TOKEN` is not also set).
- **Duplicate analysis prevention** — A second `POST /technical/{paper_id}/analyze` for a paper already being analyzed now joins the existing SSE stream instead of spawning a duplicate LLM task.
- **Concurrency cap** — At most 2 technical analyses run simultaneously (configurable). Additional requests wait for a slot, preventing simultaneous papers from exhausting the LLM rate limit.
- **SSE queue leak prevention** — Queue entries are now bounded (256 events max), reader-tracked, and reaped by a background task after 10 minutes. Client disconnects no longer produce unbounded memory accumulation.
- **Atomic SQLite cache writes** — `set_cache` now uses `INSERT … ON CONFLICT DO UPDATE` — a single atomic statement. The previous read-then-write pattern had a TOCTOU race under concurrent requests.
- **`UniqueConstraint` on generated cache** — `(cache_type, cache_key)` is now enforced at the database level. An upgrade migration deduplicates existing rows and creates the index on first startup.
- **Section ordering on restore** — Cached study sections are stored with their outline index and returned in correct order, regardless of DB insertion order.
- **`AbortController` on technical mount** — The `POST /analyze` fetch is now cancelled on component unmount, preventing double-fire in React StrictMode and on rapid tab switches.
- **`.env.example`** — Backend environment reference file (was referenced in docs but missing from the repo).
- **`frontend/.env.local.example`** — Frontend token configuration reference.
- **`CONTRIBUTING.md`** — Contributor guide covering setup, branch naming, and PR checklist.

### Changed

- `startTeaching` no longer wipes cached sections — it restores from cache if sections exist, only fetching from section 0 if genuinely starting fresh.
- Pre-fetch logic updated for sparse section arrays — checks `sections[nextIndex]` instead of `sections.length`.
- Progress bar uses `sections.filter(Boolean).length` to count only loaded sections in sparse arrays.
- `bust_technical_cache` returns HTTP 409 if an analysis is currently in progress.
- `GeneratedCache` model moved to `database.py` with cascade delete when parent paper is removed.

### Fixed

- `sendMessage` previously constructed assistant messages with ephemeral `Date.now()` IDs. Messages now sync from the database after send, ensuring IDs and timestamps match what is stored.
- Duplicate `StudyPanel` function definition removed from `page.tsx`.

---

## [1.0.0] — 2026-03-16

### Added

- Search and fetch papers from arXiv and PubMed by topic with optional date filtering
- Local PDF upload (up to 50 MB) with magic-byte validation
- Full processing pipeline: Docling extraction → chunking → sentence-transformer embedding → ChromaDB storage
- Three-pass PDF extraction: Docling (primary) → PyMuPDF (fallback) → OCR (scanned documents)
- Real-time progress streaming via Server-Sent Events; stuck papers auto-recovered on restart
- RAG chat with configurable teaching level (beginner / intermediate / advanced / expert)
- Twitter thread generation (5–8 tweets, ≤280 characters each)
- LinkedIn post generation with hook and hashtags
- LinkedIn carousel generation (6–8 slides, cover + CTA enforced)
- Carousel PDF export at 1080×1080pt in dark, light, and bold color schemes
- One-click share deeplinks to LinkedIn and Twitter/X post composers
- Configurable LLM provider: Gemini, OpenAI, Anthropic, Groq, OpenRouter
- Retry logic with exponential backoff for all LLM calls
- Prompt-injection stripping at ingestion and LLM boundaries
- System prompt credential-disclosure guard
- 255-check automated backend validation suite
- Study mode: AI-generated learning outline with section-by-section teaching and flashcards
- Technical mode: five-section structured analysis (overview, concepts, architecture, implementation, scalability)
- Generation history with copy-to-clipboard and share actions