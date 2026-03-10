# ResearchRAG — Setup & Run Guide

Follow these steps in order. The whole stack runs locally with no paid services
except a free Gemini API key.

---

## Prerequisites

| Tool | Version | Check |
|---|---|---|
| Python | 3.10+ | `python3 --version` |
| Node.js | 18+ | `node --version` |
| PostgreSQL | 14+ | `psql --version` |
| Git | any | `git --version` |

---

## Step 1 — Get a free Gemini API key

1. Go to **https://aistudio.google.com/app/apikey**
2. Sign in with a Google account
3. Click **Create API key**
4. Copy the key — you'll use it in Step 3

Free tier limits: 15 requests/min · 1,500 requests/day · 1M tokens/min

---

## Step 2 — Create the PostgreSQL database

```bash
# Connect to PostgreSQL as the postgres superuser
psql -U postgres

# Inside psql, run:
CREATE DATABASE researchrag;
\q
```

If your PostgreSQL user or password is different, update `DATABASE_URL` in Step 3.

---

## Step 3 — Backend setup

```bash
cd backend
```

### 3a. Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
```

### 3b. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, SQLAlchemy, ChromaDB, sentence-transformers, Gemini SDK,
Playwright, ReportLab, slowapi, and all other dependencies.
`sentence-transformers` will automatically download an appropriate version of PyTorch.

### 3c. Install Playwright's Chromium browser

This is required for carousel slide rendering.

```bash
playwright install chromium
```

### 3d. Set environment variables

Create a `.env` file in the `backend/` directory:

```bash
# backend/.env

# Required
GEMINI_API_KEY=your_key_here

# Optional — change if your PostgreSQL setup differs
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/researchrag

# Optional — enable API key authentication (leave empty to skip auth in dev)
# API_SECRET_KEY=change-me-to-a-long-random-string

# Optional — all other values have sensible defaults
# GEMINI_MODEL=gemini-1.5-flash
# CHROMA_DIR=./chroma_db
# DOWNLOADS_DIR=./downloads
# CAROUSEL_OUTPUT_DIR=./carousel_outputs
# MAX_EMBEDDING_BATCH_SIZE=50
```

Or export them directly:

```bash
export GEMINI_API_KEY="your_key_here"
```

### 3e. Start the backend

```bash
uvicorn app:app --reload --port 8000
```

On first startup you should see:

```
INFO | config | ✓  All required configuration present.
INFO | root   | [startup] Playwright/Chromium: OK
INFO | root   | ResearchRAG API ready.
```

If you see `Carousel UNAVAILABLE`, re-run `playwright install chromium`.

Verify the backend is running:

```bash
curl http://localhost:8000/health
# {"status":"ok","version":"3.0.0","playwright":true}
```

---

## Step 4 — Frontend setup

Open a new terminal tab.

```bash
cd frontend
```

### 4a. Install Node dependencies

```bash
npm install
```

### 4b. Create the frontend environment file

```bash
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
```

### 4c. Start the frontend

```bash
npm run dev
```

Open **http://localhost:3000** in your browser.

---

## Step 5 — Verify end-to-end

1. Type a topic in the search box (e.g. `transformer attention mechanisms`)
2. Select a source (arXiv, PubMed, or Both) and click **Search**
3. Watch the loading screen — each paper shows real-time download and processing progress
4. When processing completes, the chat view opens automatically
5. Ask a question about the paper
6. Click **Generate** to create a Twitter thread, LinkedIn post, or Carousel PDF

---

## Troubleshooting

### "GEMINI_API_KEY is not set" on startup
The app exits cleanly with this message. Set the env var and restart.

```bash
export GEMINI_API_KEY="your_key_here"
uvicorn app:app --reload --port 8000
```

### PostgreSQL connection error
Check that PostgreSQL is running and the `DATABASE_URL` is correct.

```bash
psql -U postgres -d researchrag -c "SELECT 1;"
```

### "Carousel UNAVAILABLE" in startup log
Chromium was not installed for Playwright.

```bash
playwright install chromium
```

### Embedding model slow on first use
`all-MiniLM-L6-v2` (~90MB) is downloaded from Hugging Face on the first paper.
Subsequent runs use the local cache.

### PubMed returns no results
PubMed only returns Open Access articles (the filter is applied automatically).
Try the same topic on arXiv only if PubMed comes back empty.

### Rate limit hit (HTTP 429)
The free Gemini tier allows 15 requests/minute. If you're generating content
for multiple papers quickly, wait 60 seconds and try again.
The per-IP limits on the API (5/min search, 30/min chat, 3/min generate) reset
every 60 seconds.

### Frontend can't reach backend (CORS error)
Make sure `NEXT_PUBLIC_API_URL` in `frontend/.env.local` matches the port
where the backend is running, and that `CORS_ORIGINS` in the backend env includes
`http://localhost:3000`.

---

## Running in production

1. Set `API_SECRET_KEY` to a long random string (e.g. `openssl rand -hex 32`)
2. Add `X-API-Key: <your_secret>` to frontend API calls
3. Use a process manager (`systemd`, `pm2`) instead of `--reload`
4. Put Nginx or a reverse proxy in front of both services
5. Set `CORS_ORIGINS` to your actual domain
6. Use a managed PostgreSQL instance and set `DATABASE_URL` accordingly

```bash
# Production backend (no --reload)
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

# Production frontend
npm run build
npm start
```

Note: use `--workers 1` — the in-memory `_sse_queues` and `_active_pipelines`
sets are not shared across workers. For multi-worker deployments these would
need to be moved to Redis.

---

## Quick reference

```bash
# Backend
cd backend && source .venv/bin/activate
export GEMINI_API_KEY="..."
uvicorn app:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm run dev

# Health check
curl http://localhost:8000/health

# View logs
# Backend logs appear in the terminal where uvicorn is running
# Frontend logs appear in the browser console and the npm terminal
```
