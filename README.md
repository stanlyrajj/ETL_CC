# ResearchRAG

ResearchRAG is a local tool for solo educational content creators who want to understand academic research papers and turn them into social media content. It fetches papers from arXiv and PubMed, processes them through a RAG pipeline, lets you chat with them at any knowledge level, and generates ready-to-share LinkedIn posts, Twitter threads, and carousel PDFs — all running on your own machine with no hosted backend.

## Features

- **Search and fetch** papers from arXiv and PubMed by topic, with optional date filtering
- **Upload** your own PDFs directly
- **Real-time processing** with live stage updates via Server-Sent Events
- **Chat** with any paper at beginner, intermediate, or advanced level
- **Generate** Twitter threads, LinkedIn posts, and LinkedIn carousel slides
- **Export** carousels as PDF (1080×1080pt, three color schemes)
- **Share** directly to platform composers with one click via deep-links
- **Configurable LLM** — switch between Gemini, OpenAI, and Anthropic with one env var

## Quick start

See [SETUP.md](SETUP.md) for complete step-by-step instructions.

```bash
# Backend
cd backend && pip install -r requirements.txt
cp .env.example .env   # then fill in LLM_PROVIDER and LLM_API_KEY
uvicorn main:app --reload --port 8000

# Frontend (new terminal)
cd frontend && npm install && npm run dev
```

Open **http://localhost:3000**.

## Workflow

```
Search or Upload → Processing → Chat → Generate → Export / Share
```

1. **Search or upload** — enter a topic or upload a PDF. Papers are fetched and stored.
2. **Processing** — each paper is downloaded, extracted, chunked, embedded, and indexed. Progress is streamed live.
3. **Chat** — ask questions about any processed paper. Adjust the teaching level at any time.
4. **Generate** — produce Twitter threads, LinkedIn posts, or carousel slides from the paper.
5. **Export / Share** — download carousel PDFs or open the platform composer pre-filled with your content.

## Supported LLM providers

| Provider  | Env value   | Default model              |
|-----------|-------------|----------------------------|
| Google    | `gemini`    | `gemini-2.0-flash`         |
| OpenAI    | `openai`    | `gpt-4o-mini`              |
| Anthropic | `anthropic` | `claude-sonnet-4-20250514` |

Set `LLM_PROVIDER` and `LLM_API_KEY` in `backend/.env`.

## Supported data sources

- **arXiv** — via the official `arxiv` Python SDK
- **PubMed** — via NCBI E-utilities (open-access filter applied automatically)
- **Local PDF** — upload any PDF up to 50 MB

## Supported output platforms

- **Twitter / X** — thread of up to 8 tweets, each ≤ 280 characters
- **LinkedIn** — single post with hook and hashtags
- **LinkedIn carousel** — 6–8 slides exported as a 1080×1080pt PDF (dark, light, or bold color scheme)

## Extending the project

See [docs/extending.md](docs/extending.md) for instructions on adding new data sources or output platforms.

See [docs/architecture.md](docs/architecture.md) for a full description of the system design.

## License

MIT
