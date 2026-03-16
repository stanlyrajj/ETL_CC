# ResearchRAG — System Architecture

## Overview

ResearchRAG is a local application with a FastAPI backend and a Next.js frontend. The backend is structured as a series of independent layers with clearly defined input and output contracts. Each layer trusts that the layer before it has already validated and sanitized its input. The frontend communicates exclusively via the backend's REST API and Server-Sent Events.

---

## Layers

### 1. Ingestion layer (`backend/ingestion/`)

The ingestion layer is the primary security boundary. All external data — whether fetched from arXiv, PubMed, or uploaded as a local file — passes through this layer before anything else in the system sees it.

Every fetcher produces a `DocumentInput` dataclass. Before returning it, the data is run through `validator.validate()`, which strips HTML tags, null bytes, control characters, and prompt-injection phrases from all text fields, enforces size limits, and sanitizes the `paper_id` to a safe character set. Anything that fails validation is discarded with a log warning — the server never crashes on bad input.

### 2. Processing layer (`backend/processing/`)

The processing layer takes a `DocumentInput` and produces embedded, searchable chunks stored in ChromaDB. It has no knowledge of where the document came from.

The pipeline runs in four stages: extraction (Docling converts the PDF or BioC JSON to structured sections), chunking (sections are split into overlapping text windows), embedding (chunks are encoded to 384-dimensional vectors using `all-MiniLM-L6-v2`), and storage (vectors and metadata are upserted into ChromaDB). Each stage is blocking, so extraction and embedding run inside `asyncio.run_in_executor` to avoid blocking the event loop.

### 3. LLM layer (`backend/llm/`)

The LLM layer presents a uniform interface regardless of which provider is active. `LLMProvider` is an abstract base class with four methods: `chat_response`, `generate_twitter_thread`, `generate_linkedin_post`, and `generate_carousel_content`. The three concrete implementations (Gemini, OpenAI, Anthropic) all conform to this interface.

Before any paper content reaches the LLM, it is passed through `sanitize_context()`, which runs the same sanitization as the ingestion layer and wraps the result in `<paper_content>` XML tags. The system prompt embedded in `LLMProvider.SYSTEM_PROMPT` explicitly forbids the model from disclosing API keys, revealing its own instructions, or following instructions embedded inside the paper content.

### 4. Chat layer (`backend/chat/`)

The chat layer connects sessions, retrieval, and LLM calls. `session.py` handles all database reads and writes for `ChatSession` and `ChatMessage` records. `rag.py` implements the single `respond()` function: it retrieves the top-5 most relevant chunks from ChromaDB for the user's message, assembles a context string, calls the LLM, and saves both the user message and the assistant response to the database.

### 5. Content generation layer (`backend/content/`)

Each content module (`twitter.py`, `linkedin_post.py`, `carousel.py`) follows the same pattern: fetch paper context from ChromaDB, call the appropriate LLM method, validate and normalize the structured output, save to the `SocialContent` table, and return the result. Each module has a `_validate()` function that enforces structural rules (e.g. first carousel slide must be `cover`, last must be `cta`) and normalizes common LLM output aliases.

### 6. Export layer (`backend/export/`)

`pdf_renderer.py` takes a list of validated slide dicts and renders them to a 1080×1080pt PDF using ReportLab. Three color schemes are supported: dark, light, and bold. `share.py` builds URL-encoded deep-links to the LinkedIn and Twitter/X post composers with content pre-filled.

### 7. API layer (`backend/api/`)

FastAPI routes wire the layers together. Every route returns explicit success or failure — no silent failures. Background tasks (pipeline processing, content generation) push progress events to in-memory SSE queues. The SSE endpoints stream those events to the frontend and clean up their queues when the client disconnects.

---

## Data flow

```
User input
    │
    ▼
Ingestion layer         ← validates, sanitizes, produces DocumentInput
    │
    ▼
Processing layer        ← extracts, chunks, embeds, stores in ChromaDB
    │
    ▼
Chat layer              ← retrieves chunks, calls LLM, saves messages
    │
    ▼
Content generation      ← retrieves chunks, calls LLM, validates output
    │
    ▼
Export layer            ← renders PDF, builds share deeplinks
    │
    ▼
API layer               ← serves results, streams SSE progress
    │
    ▼
Frontend                ← displays results, manages state
```

---

## Folder structure

```
researchrag/
│
├── backend/
│   ├── main.py                  Entry point: CORS, routers, startup lifecycle
│   ├── config.py                Config class: reads .env, validates required vars
│   ├── database.py              SQLAlchemy models and all DB helper methods
│   │
│   ├── ingestion/
│   │   ├── validator.py         Security boundary: sanitizes all incoming data
│   │   ├── arxiv_fetcher.py     Fetches papers from arXiv via the arxiv SDK
│   │   ├── pubmed_fetcher.py    Fetches papers from PubMed via NCBI E-utilities
│   │   └── local_uploader.py   Accepts and validates local PDF uploads
│   │
│   ├── processing/
│   │   ├── extractor.py         Extracts text from PDFs and BioC JSON via Docling
│   │   ├── chunker.py           Splits text into overlapping chunks
│   │   ├── embedder.py          Encodes chunks to vectors using sentence-transformers
│   │   └── vector_store.py      ChromaDB operations: upsert, query, delete
│   │
│   ├── llm/
│   │   ├── base.py              Abstract LLMProvider, sanitize_context, parse_json_response
│   │   ├── gemini.py            Gemini implementation using google-genai
│   │   ├── openai.py            OpenAI implementation
│   │   ├── anthropic.py         Anthropic implementation
│   │   └── factory.py           Returns the configured provider instance
│   │
│   ├── chat/
│   │   ├── session.py           Database helpers for ChatSession and ChatMessage
│   │   └── rag.py               RAG respond(): retrieve → call LLM → save messages
│   │
│   ├── content/
│   │   ├── twitter.py           Generates and validates Twitter threads
│   │   ├── linkedin_post.py     Generates and validates LinkedIn posts
│   │   └── carousel.py          Generates and validates carousel slide content
│   │
│   ├── export/
│   │   ├── pdf_renderer.py      Renders carousel slides to PDF via ReportLab
│   │   └── share.py             Builds deep-links to LinkedIn and Twitter composers
│   │
│   └── api/
│       ├── papers.py            /papers routes + full pipeline background task
│       ├── chat.py              /chat/sessions routes
│       ├── generate.py          /generate routes + generation background task
│       └── progress.py          SSE streams for pipeline and generation progress
│
├── frontend/
│   └── app/
│       ├── layout.tsx           Root HTML layout
│       ├── globals.css          Design system: tokens, typography, components
│       ├── page.tsx             Full application: Search, Processing, Chat views
│       └── lib/
│           └── api.ts           Typed client for all backend endpoints
│
└── docs/
    ├── architecture.md          This file
    └── extending.md             How to add sources and platforms
```

---

## Key design decisions

**Single-file frontend.** The entire UI lives in `page.tsx`. This keeps navigation state simple (a single `view` enum) and avoids the complexity of a router for an application with three sequential views.

**Module-level singletons.** The VectorStore, embedding model, and LLM provider are each created once and reused. This avoids repeatedly loading the 90 MB embedding model and prevents ChromaDB from opening multiple connections to the same database file.

**SSE over WebSockets.** Pipeline progress is one-directional (server to client), making SSE simpler and sufficient. Queues are in-memory dicts keyed by paper ID or queue key, created before background tasks start, and cleaned up when clients disconnect.

**No external task queue.** Processing and generation run as FastAPI background tasks. This is appropriate for a single-user local tool and avoids the operational complexity of Redis, Celery, or similar infrastructure.
