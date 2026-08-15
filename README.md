# RoleRead — Career Intelligence Assistant

RoleRead is a small, self-contained web app that lets you upload your resume and one or more job descriptions, then chat with an AI about fit, skill gaps, and interview prep.

## Quick start

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

uvicorn app.main:app --reload --port 8420
```

Open `http://127.0.0.1:8420/`. FastAPI serves the frontend as static files, so no separate frontend server is needed.

## How to use

1. **Upload your resume** (PDF, TXT, or MD).
2. **Upload one or more job descriptions** (PDF, TXT, or MD).
3. **Ask questions**, for example:
   - "What skills am I missing for Job #1?"
   - "How does my experience align with Job #2?"
   - "Help me prepare for an interview for Job #1"

## Stack

- **Backend:** FastAPI
- **Vector store:** Chroma (embedded, no Docker)
- **Embeddings / LLM:** OpenAI (`text-embedding-3-small`, `gpt-4o-mini`)
- **Chunking:** LangChain `RecursiveCharacterTextSplitter`
- **Frontend:** vanilla HTML, CSS, JS
- **Isolation:** per-session Chroma collections based on `X-Session-Id`

## Notes

- `backend/.env` is gitignored — never commit your API key.
- Chroma data lives in `backend/chroma_data/` and is gitignored.
