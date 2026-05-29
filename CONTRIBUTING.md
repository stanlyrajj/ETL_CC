# Contributing to ResearchRAG

Thank you for your interest. This document covers how to set up a development environment, the branch and PR conventions used in this project, and the checklist before submitting.

---

## Development setup

Follow [SETUP.md](SETUP.md) to get the project running. For development specifically:

```bash
# Backend — run with auto-reload
uvicorn main:app --reload --port 8000

# Frontend — run with hot reload
npm run dev
```

Run the validation suite before any PR:

```bash
cd backend
python validate_backend.py --api
```

All 255 checks must pass.

---

## Branch naming

| Type | Pattern | Example |
|---|---|---|
| Feature | `feat/short-description` | `feat/semantic-scholar-source` |
| Bug fix | `fix/short-description` | `fix/study-section-ordering` |
| Docs | `docs/short-description` | `docs/add-groq-setup-notes` |
| Refactor | `refactor/short-description` | `refactor/split-page-components` |

---

## PR checklist

Before opening a pull request:

- [ ] `python validate_backend.py --api` passes all 255 checks
- [ ] New backend endpoints have a corresponding test in `validate_backend.py`
- [ ] New environment variables are documented in `.env.example`
- [ ] `CHANGELOG.md` has an entry under `[Unreleased]`
- [ ] No API keys, local paths, or personal data in any committed file
- [ ] `requirements.txt` is updated if new Python packages are added
- [ ] `package.json` is updated if new npm packages are added

---

## Adding a new LLM provider

1. Create `backend/llm/your_provider.py` implementing `LLMProvider` from `base.py`
2. Register it in `backend/llm/factory.py`
3. Add the provider name to the validation list in `config.py`
4. Document the provider and its default model in `README.md` and `.env.example`

## Adding a new data source or output platform

See [docs/extending.md](docs/extending.md) for step-by-step instructions.

---

## Code style

- Python: follow the existing style (no formatter is enforced, but keep line length under 120 characters)
- TypeScript: no linter is enforced, but match the existing inline style patterns in `page.tsx`
- Imports: group standard library, third-party, and local imports with a blank line between each group
- Comments: prefer explaining *why* over *what*

---

## Reporting bugs

Open a GitHub issue with:
- The exact error message or unexpected behavior
- The steps to reproduce
- Your OS, Python version, Node.js version, and LLM provider
- Relevant lines from the uvicorn or browser console logs
