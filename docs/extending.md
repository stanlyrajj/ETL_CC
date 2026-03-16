# Extending ResearchRAG

## Adding a new data source

Data sources live in `backend/ingestion/`. Each source is a single Python file that exposes a `search()` async function returning a list of validated `DocumentInput` objects.

### Step 1 — Create the fetcher file

Create `backend/ingestion/your_source_fetcher.py`. The file must implement:

```python
async def search(
    topic: str,
    limit: int,
    date_from: datetime | None = None,
) -> list[DocumentInput]:
    ...
```

The function must:

1. Fetch results from the external source
2. Map each result to a plain `dict` with these fields:
   - `paper_id` — a unique string identifier for this paper
   - `source` — a short string name for your source (e.g. `"semanticscholar"`)
   - `title` — paper title
   - `abstract` — paper abstract (may be empty string)
   - `authors` — list of author name strings
   - `url` — link to the paper
   - `file_path` — path to a local PDF if already downloaded, otherwise empty string
   - `topic` — the search topic passed in
   - `extra_metadata` — dict of any additional fields you want to store
3. Pass each dict through `validator.validate(raw_dict)` before adding to the output
4. Skip any paper that raises `ValidationError` — log a warning and continue
5. Respect a rate limit between requests (add your rate limit constant to `config.py`)

Look at `arxiv_fetcher.py` and `pubmed_fetcher.py` for reference implementations.

### Step 2 — Register it in the search route

Open `backend/api/papers.py` and update the `search_papers` route:

```python
# Add your import at the top
from ingestion import your_source_fetcher

# Inside search_papers(), add a new branch:
if source in ("yoursource", "both"):
    your_docs = await your_source_fetcher.search(request.topic, request.limit, date_from)
    docs.extend(your_docs)
```

Update the source validation check to include your new source name:

```python
if source not in ("arxiv", "pubmed", "yoursource", "both"):
    raise HTTPException(status_code=400, ...)
```

### Step 3 — Add the rate limit to config

In `backend/config.py`, add a new config variable in `__init__`:

```python
self.YOUR_SOURCE_RATE_LIMIT: float = float(os.getenv("YOUR_SOURCE_RATE_LIMIT", "1.0"))
```

And add a corresponding entry to `backend/.env.example`.

---

## Adding a new output platform

Output platforms live in `backend/content/`. Each platform is a single Python file with a `generate()` async function and a `_validate()` function.

### Step 1 — Create the content module

Create `backend/content/your_platform.py`. The file must implement:

```python
async def generate(
    paper_id: str,
    style: str,
    tone: str,
    # add any platform-specific params here
) -> dict:
    ...
```

The function must:

1. Retrieve paper context from the VectorStore using `get_paper_context(paper_id, max_chars=4000)`. If the context is empty, raise `ValueError` with a clear message.
2. Fetch the paper title from the database.
3. Call the appropriate LLM method on `get_provider()`. If no suitable method exists on `LLMProvider`, add one (see Step 3 below).
4. Pass the raw LLM output through a `_validate()` function that checks structure and normalizes field aliases. Raise a clear error if required fields are missing.
5. Save the result to the `SocialContent` table via `db.save_social()` with `platform="yourplatform"`.
6. Return the validated result dict.

Look at `twitter.py` or `linkedin_post.py` for a minimal reference, or `carousel.py` for a more complex example with structural validation.

### Step 2 — Register it in the generate route

Open `backend/api/generate.py` and add your platform:

```python
# Add import at the top
from content.your_platform import generate as your_platform_generate

# Add to _VALID_PLATFORMS
_VALID_PLATFORMS = ("twitter", "linkedin", "carousel", "yourplatform")

# Add a branch in _run_generate()
elif platform == "yourplatform":
    result = await your_platform_generate(request.paper_id, request.style, request.tone)
```

### Step 3 — Add an LLM method (if needed)

If your platform needs a new kind of LLM output, add an abstract method to `backend/llm/base.py`:

```python
@abstractmethod
async def generate_your_platform(
    self,
    context: str,
    title: str,
    style: str,
    tone: str,
) -> dict:
    """Generate content for YourPlatform. Returns structured dict."""
```

Then implement it in all three provider files: `gemini.py`, `openai.py`, and `anthropic.py`. Each implementation must:

- Call `sanitize_context(context)` before including context in any prompt
- Use `parse_json_response()` to parse the LLM output
- Never include the API key, file paths, or config values in any prompt string

### Step 4 — Add the platform to the frontend

Open `frontend/app/page.tsx` and update the `Platform` type and the platform selector buttons:

```typescript
type Platform = 'twitter' | 'linkedin' | 'carousel' | 'yourplatform'
```

In the `GenerationPanel` component, add `'yourplatform'` to the platform buttons array and add a `renderPreview()` branch for your content format.
