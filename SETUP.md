# ResearchRAG — Setup Guide

## Prerequisites

- Python 3.11 or higher
- Node.js 18 or higher
- An API key for at least one LLM provider (Gemini, OpenAI, or Anthropic)
- 4 GB of free disk space (for Docling and embedding models, downloaded on first run)

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

# Windows
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set the two required values:

```
LLM_PROVIDER=gemini          # or: openai, anthropic
LLM_API_KEY=your_key_here
```

All other variables have sensible defaults and do not need to be changed for a standard setup. See `.env.example` for the full list with descriptions.

### Pre-download Docling models (recommended)

Docling downloads its layout and OCR models on first use (~1–2 GB). Running this once upfront avoids a long wait during your first paper processing:

```bash
python -c "from docling.document_converter import DocumentConverter; DocumentConverter(); print('Models ready.')"
```

### Start the backend

```bash
uvicorn main:app --reload --port 8000
```

You should see:

```
INFO: ResearchRAG startup complete.
```

---

## 3. Frontend setup

Open a new terminal.

```bash
cd frontend
npm install
npm run dev
```

The frontend starts at **http://localhost:3000**.

---

## 4. Verify the installation

With both servers running, open http://localhost:3000 in your browser. You should see the ResearchRAG search interface.

To run the backend validation suite:

```bash
cd backend
python validate_backend.py --api
```

All 255 checks should pass.

---

## Troubleshooting

### `LLM_PROVIDER is not set` on startup
You have not created a `.env` file, or it is in the wrong location. The file must be at `backend/.env` (not the project root).

### `ModuleNotFoundError` when starting the backend
Your virtual environment is not activated. Run `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows) before starting uvicorn.

### `Form data requires "python-multipart"` error
Run `pip install python-multipart` and restart the server.

### Docling extracts 0 characters from a PDF
Docling's models have not finished downloading. Run the pre-download command from Step 2 and wait for it to complete before processing PDFs.

### ChromaDB permission error on Windows
Windows does not support symlinks without Developer Mode. This shows as a warning but does not prevent ChromaDB from working. Enable Developer Mode or run the terminal as Administrator to silence the warning.

### Frontend shows a blank page or `fetch` errors
Confirm the backend is running on port 8000. The `next.config.js` proxy rewrites `/api/*` to `http://localhost:8000/api/*` — if your backend is on a different port, update this file.

### `CORS` errors in the browser console
Add your frontend origin to `CORS_ORIGINS` in `backend/.env`:
```
CORS_ORIGINS=http://localhost:3000
```
Restart the backend after changing this value.

### Papers stuck in `processing` stage after a server restart
This is handled automatically. On startup, the backend resets all papers stuck in intermediate stages back to `pending` and re-queues them for processing.
