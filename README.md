<p align="center">
  <img src="docs/assets/banner.png" alt="ResearchRAG" width="100%" />
</p>

<h1 align="center">ResearchRAG</h1>

<p align="center">
  A fully local research-to-content pipeline — fetch academic papers, chat with them, and turn them into Twitter threads, LinkedIn posts, and carousel PDFs.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" />
  <img alt="Next.js" src="https://img.shields.io/badge/Next.js-15-000000?logo=nextdotjs&logoColor=white" />
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green" />
  <img alt="Local first" src="https://img.shields.io/badge/runs-100%25%20locally-blueviolet" />
</p>

---

## What it does

ResearchRAG is a desktop tool for researchers and content creators who regularly work with academic papers. It handles the full workflow in one place:

```
Search / Upload → Process → Chat → Study → Technical → Generate → Export
```

| Step | What happens |
|---|---|
| **Search** | Fetch papers from arXiv or PubMed by topic, or upload your own PDF |
| **Process** | Papers are extracted, chunked, embedded, and indexed — entirely on your machine |
| **Chat** | Ask questions about any paper at beginner, intermediate, or expert level |
| **Study** | Step through an AI-generated learning plan with flashcards; resumes exactly where you left off |
| **Technical** | Five-section deep dive — architecture, concepts, implementation, trade-offs |
| **Generate** | Produce Twitter threads, LinkedIn posts, and carousel slides |
| **Export** | Download carousel PDFs or open the platform composer pre-filled |

Everything runs locally. No cloud backend, no telemetry, no data leaves your system except the LLM API call (your key, your choice of provider).

---

## Screenshot

> _Add a screenshot or screen recording of the app here._

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Python 3.11, fully async |
| PDF extraction | Docling (layout-aware) → PyMuPDF (fast fallback) → OCR (scanned) |
| Embedding | `sentence-transformers` · `all-MiniLM-L6-v2` · CPU-only |
| Vector store | ChromaDB (embedded, no server required) |
| Database | SQLite + SQLAlchemy async |
| LLM providers | OpenAI · Anthropic · Google Gemini · Groq · OpenRouter |
| Frontend | Next.js 15 + TypeScript |
| PDF export | ReportLab (1080×1080pt carousel slides) |

---

## Features

- **Multi-source ingestion** — arXiv, PubMed, local PDF (up to 50 MB)
- **Three-pass extraction** — Docling → PyMuPDF → OCR fallback chain handles any PDF
- **RAG chat** — retrieval-augmented answers grounded in the actual paper content
- **Study mode** — AI-generated learning plan, section-by-section teaching, flashcards; cached in SQLite and fully restorable on revisit
- **Technical mode** — structured five-section breakdown cached per paper, instant on return
- **Resizable panels** — drag to resize the generate panel and chat input bar
- **Content generation** — Twitter threads, LinkedIn posts, 6–8 slide carousels
- **PDF export** — dark, light, and bold color schemes at pixel-accurate 1080×1080pt
- **One-click share** — deep-links to LinkedIn and Twitter/X composers with content pre-filled
- **Security** — token auth, loopback-only binding, prompt-injection stripping, magic-byte validation
- **Resilient** — stuck papers auto-recovered on restart, atomic SQLite upserts, SSE memory leak prevention

---

## Quick start

### Prerequisites

- Python 3.11+
- Node.js 18+
- API key for at least one supported LLM provider

### 1. Clone

```bash
git clone https://github.com/your-username/researchrag.git
cd researchrag
```

### 2. Backend

```bash
cd backend

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env            # then open .env and set LLM_PROVIDER + LLM_API_KEY

uvicorn main:app --reload --port 8000
```

### 3. Frontend

```bash
# open a new terminal
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**.

> **First run note:** Docling downloads its layout and OCR models (~1–2 GB) on first paper processing. Pre-download once to avoid the wait:
> ```bash
> python -c "from docling.document_converter import DocumentConverter; DocumentConverter(); print('ready')"
> ```

Full setup guide with troubleshooting → [SETUP.md](SETUP.md)

---

## LLM providers

| Provider | `LLM_PROVIDER` | Default model |
|---|---|---|
| Google Gemini | `gemini` | `gemini-2.0-flash` |
| OpenAI | `openai` | `gpt-4o-mini` |
| Anthropic | `anthropic` | `claude-sonnet-4-20250514` |
| Groq | `groq` | `llama-3.3-70b-versatile` |
| OpenRouter | `openrouter` | set via `LLM_MODEL` |

Switch provider by changing `LLM_PROVIDER` in `backend/.env` and restarting. No code changes needed.

---

## Project structure

```
researchrag/
├── backend/
│   ├── main.py                   Entry point — CORS, auth middleware, routers, startup
│   ├── config.py                 Environment config with validation and defaults
│   ├── database.py               SQLAlchemy models + async DB helpers
│   ├── ingestion/                arXiv · PubMed · local PDF — all sanitized at boundary
│   ├── processing/               Extract → chunk → embed → vector store pipeline
│   ├── llm/                      Provider-agnostic LLM abstraction + rate limiter
│   ├── chat/                     RAG sessions, retrieval, message persistence
│   ├── content/                  Twitter · LinkedIn · carousel generators
│   ├── export/                   PDF renderer + share deeplink builder
│   ├── api/                      FastAPI routes + SSE progress streams
│   └── validate_backend.py       255-check automated test suite
├── frontend/
│   └── app/
│       ├── page.tsx              Full application UI (single-file, intentional)
│       ├── globals.css           Design system — tokens, typography, components
│       └── lib/api.ts            Typed API client for all backend endpoints
└── docs/
    ├── architecture.md           System design, layer contracts, data flow
    └── extending.md              How to add new data sources and output platforms
```

---

## Security model

| Concern | How it is addressed |
|---|---|
| Network exposure | Binds to `127.0.0.1` by default — not reachable over a network |
| API access control | Optional `APP_TOKEN` — all endpoints require `X-App-Token` header when set |
| Prompt injection | Stripped at ingestion boundary and again at LLM call boundary |
| File upload safety | Magic-byte validation — file type checked by content, not extension |
| Credential leakage | System prompt explicitly prohibits model from disclosing config or keys |
| Concurrent writes | Atomic `INSERT … ON CONFLICT DO UPDATE` — no TOCTOU race on cache writes |

---

## How it is built — key decisions

**Local-first.** All data (papers, embeddings, chat history, generated content) lives in SQLite and ChromaDB on your machine. The only external calls are to the LLM API of your choice.

**Provider-agnostic LLM layer.** `LLMProvider` is an abstract base class. Switching from OpenAI to Gemini is a one-line `.env` change.

**Three-pass PDF extraction.** Docling handles complex academic layouts. PyMuPDF runs if Docling fails or times out. A final OCR pass handles scanned documents. This makes the pipeline resilient across every paper format encountered in the wild.

**SSE for progress streaming.** Pipeline and generation progress is streamed live to the browser via Server-Sent Events. Queues are bounded, reader-tracked, and reaped by a background task — no memory leak on client disconnect.

**Study and technical content cached in SQLite.** Generated content is stored per paper and served instantly on revisit. A regenerate button is the only way to clear it.

---

## Extending

See [docs/extending.md](docs/extending.md) for step-by-step instructions on adding new data sources (e.g. Semantic Scholar) or output platforms (e.g. Bluesky threads).

---

## License

MIT — see [LICENSE](LICENSE).