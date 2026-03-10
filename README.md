# ResearchRAG

**Discover open-access research papers → chat with them → turn them into LinkedIn carousels, Twitter threads, and social posts.**

ResearchRAG searches arXiv and PubMed, processes papers through a full RAG pipeline, and opens a conversational teaching interface powered by Gemini. Every session lives in a sidebar so you can revisit any paper later. A single "Generate" button turns a processed paper into ready-to-post social content.

---

## What it does

```
Topic search ──► arXiv + PubMed APIs
                       │
                  Download PDFs / BioC JSON
                       │
             Extract → Chunk → Embed (all-MiniLM-L6-v2)
                       │
              ChromaDB (vector store) + PostgreSQL
                       │
              ┌────────┴────────┐
         RAG Chat           Generate
    (Gemini 1.5 Flash)   Twitter / LinkedIn / Carousel PDF
```

- **Multi-session sidebar** — each topic is a separate chat session, persisted to PostgreSQL
- **Level selector** — Beginner / Intermediate / Advanced, switchable mid-conversation
- **Real-time progress** — Server-Sent Events stream per paper as it moves through the pipeline
- **Carousel renderer** — Playwright renders Emerald-palette slides → PNG → PDF download
- **Fully free** — Gemini 1.5 Flash (free tier), all-MiniLM-L6-v2 runs locally, arXiv + PubMed are open-access

---

## Project layout

```
researchrag/
├── backend/
│   ├── app.py                  # FastAPI — all routes in one file
│   ├── config.py               # Central config, validated at startup
│   ├── database.py             # SQLAlchemy async models + CRUD
│   ├── gemini_client.py        # Gemini API wrapper (async, injection-hardened)
│   ├── processor.py            # PDF/JSON extract → chunk → embed → ChromaDB
│   ├── carousel_renderer.py    # Playwright slide renderer → PDF
│   └── requirements.txt
└── frontend/
    └── app/
        ├── page.tsx            # Single-page app (search / loading / chat views)
        └── globals.css         # Brilliant Emerald design system
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, React 18, TypeScript |
| Backend | FastAPI 0.111, Python 3.10+ |
| Database | PostgreSQL 14+ (SQLAlchemy 2.0 async) |
| Vector store | ChromaDB 0.5 (cosine similarity) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers (local, free) |
| LLM | Google Gemini 1.5 Flash (free tier — 15 RPM / 1500 RPD) |
| Carousel rendering | Playwright (Chromium) → ReportLab PDF |
| Rate limiting | slowapi per-IP (5/min search, 30/min chat, 3/min generate) |

---

## API overview

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/papers/search` | Search + start pipeline. Rate-limited 5/min. |
| GET | `/api/papers/{id}/progress` | SSE stream — pipeline stages in real time |
| GET | `/api/papers` | List all papers |
| DELETE | `/api/papers/{id}` | Delete paper + vectors |
| GET | `/api/chat/sessions` | All sessions (sidebar data) |
| GET | `/api/chat/sessions/{id}` | Session + full message history |
| POST | `/api/chat/sessions/{id}/message` | Send message, get Gemini reply. Rate-limited 30/min. |
| POST | `/api/generate` | Queue content generation |
| GET | `/api/generate/{key}/progress` | SSE stream for generation progress |
| GET | `/api/generate/history/{id}` | Past generations for a paper |
| POST | `/api/query` | RAG semantic search |
| GET | `/health` | Health + Playwright status |

---

## Environment variables

All have sensible defaults except `GEMINI_API_KEY`.

| Variable | Default | Required |
|---|---|---|
| `GEMINI_API_KEY` | — | **Yes** |
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/researchrag` | No |
| `API_SECRET_KEY` | `""` (auth disabled) | No |
| `GEMINI_MODEL` | `gemini-1.5-flash` | No |
| `CHROMA_DIR` | `./chroma_db` | No |
| `DOWNLOADS_DIR` | `./downloads` | No |
| `CAROUSEL_OUTPUT_DIR` | `./carousel_outputs` | No |
| `ARXIV_RATE_LIMIT` | `3.0` (seconds) | No |
| `PUBMED_RATE_LIMIT` | `0.4` (seconds) | No |
| `CHUNK_SIZE` | `1000` (chars) | No |
| `CHUNK_OVERLAP` | `200` (chars) | No |
| `MAX_EMBEDDING_BATCH_SIZE` | `50` | No |
| `MAX_RETRIES` | `3` | No |
| `CORS_ORIGINS` | `http://localhost:3000` | No |

Set `API_SECRET_KEY` to a long random string in production. When set, every endpoint except `/health` requires an `X-API-Key` header matching that value. Leave it empty to run without auth (dev mode).

---

## Security notes (v3)

- **Prompt injection hardening** — paper content is sanitized and wrapped in `<paper_content>` XML delimiters before entering the Gemini prompt
- **API key never logged** — a log filter redacts `GEMINI_API_KEY` from all log records
- **Path traversal prevention** — `paper_id` is sanitized with `_safe_filename()` before any filesystem use
- **Rate limiting** — slowapi per-IP limits prevent Gemini free-tier quota exhaustion
- **XSS-safe carousel** — `html.escape(quote=True)` used in all Playwright-rendered HTML

---

## Known limitations

- Scanned/image PDFs fail with an explicit error — OCR is not implemented
- PubMed results must be Open Access (filter applied automatically)
- Gemini free tier: 15 RPM / 1,500 requests/day — heavy use will hit limits
- No user accounts — all sessions are global (single-user deployment assumed)
- Carousel PDFs are stored on disk, not in cloud storage
