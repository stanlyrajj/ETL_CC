# ResearchRAG — Setup Guide

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | 3.12 and 3.13 also work |
| Node.js | 18+ | 20 LTS recommended |
| LLM API key | — | One of: Gemini, OpenAI, Anthropic, Groq, OpenRouter |
| Disk space | ~4 GB | For Docling models and embedding model (downloaded once) |
| RAM | 4 GB+ | 8 GB recommended when processing large papers |

---

## 1. Clone the repository

```bash
git clone https://github.com/your-username/researchrag.git
cd researchrag
```

---

## 2. Backend setup

### Create and activate a virtual environment

```bash
cd backend
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, SQLAlchemy, ChromaDB, Docling, sentence-transformers, and all LLM provider SDKs. Expect 2–5 minutes on first install.

### Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set the two required values:

```env
LLM_PROVIDER=gemini          # or: openai | anthropic | groq | openrouter
LLM_API_KEY=your_key_here
```

All other values have sensible defaults. See `.env.example` for the full reference.

### Pre-download Docling models (recommended)

Docling downloads its layout and OCR models (~1–2 GB) on first use. Running this once avoids a long wait during your first paper:

```bash
python -c "from docling.document_converter import DocumentConverter; DocumentConverter(); print('Models ready.')"
```

### Start the backend

```bash
uvicorn main:app --reload --port 8000
```

A successful start looks like:

```
INFO: Config validated. Directories ready.
INFO: Database initialised.
INFO: ResearchRAG startup complete.
INFO: Application startup complete.
```

---

## 3. Frontend setup

Open a **new terminal** (keep the backend running).

```bash
cd frontend
npm install
npm run dev
```

The frontend starts at **http://localhost:3000**.

---

## 4. Verify the installation

With both servers running, open http://localhost:3000. You should see the ResearchRAG search screen.

To run the backend validation suite:

```bash
cd backend
python validate_backend.py          # static checks (no server needed)
python validate_backend.py --api    # full API checks (server must be running)
```

All 255 checks should pass.

---

## Optional: Token authentication

If you want to protect the API (e.g. running on a shared machine or with `HOST=0.0.0.0`):

```bash
# Generate a token
python -c "import secrets; print(secrets.token_hex(32))"
```

Add to `backend/.env`:
```env
APP_TOKEN=your_generated_token_here
```

Create `frontend/.env.local` (copy from `frontend/.env.local.example`):
```env
NEXT_PUBLIC_APP_TOKEN=your_generated_token_here
```

Restart both servers. All API requests will now require the `X-App-Token` header, which the frontend sends automatically.

---

## Troubleshooting

### `LLM_PROVIDER is not set` on startup
The `.env` file is missing or in the wrong location. It must be at `backend/.env`.

### `ModuleNotFoundError` when starting the backend
Your virtual environment is not activated. Run the activate command from Step 2 before starting uvicorn.

### `Form data requires "python-multipart"` error
```bash
pip install python-multipart
```

### Docling extracts 0 characters from a PDF
The Docling models have not finished downloading. Run the pre-download command from Step 2 and wait for completion before processing papers.

### `CERTIFICATE_VERIFY_FAILED` on macOS
```bash
/Applications/Python\ 3.x/Install\ Certificates.command
```

### ChromaDB permission error on Windows
Windows requires Developer Mode for symlinks. Enable it in Settings → Developer Options, or run the terminal as Administrator.

### Frontend shows a blank page or fetch errors
Confirm the backend is running on port 8000. The `next.config.js` proxy rewrites `/api/*` to `http://localhost:8000/api/*`. If your backend is on a different port, update `next.config.js` and restart the frontend.

### Papers stuck in `processing` after a server restart
This is handled automatically. The backend resets all stuck papers to `pending` on startup and re-queues them.

### Rate limit errors (429) from your LLM provider
The backend has a built-in rate limiter with exponential backoff. If you are processing many papers simultaneously, reduce the number of concurrent operations or upgrade your API tier.