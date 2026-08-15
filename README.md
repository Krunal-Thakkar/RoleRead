# RoleRead — Career Intelligence Assistant

A RAG-based assistant that analyzes a resume against one or more job descriptions and answers
questions about fit, skill gaps, experience alignment, and interview preparation.

Built for the take-home task in `Task.docx` (Option 4). Full design reasoning and trade-offs are
in [`PLAN.md`](./PLAN.md) — this README summarizes the "what" and "why" for anyone reviewing the
final result; PLAN.md has the day-of decision-making detail.

---

## Quick setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

uvicorn app.main:app --reload --port 8420
```

Open `http://127.0.0.1:8420/` — the FastAPI app serves the frontend directly as static files, so
there's nothing else to run. No Docker, no separate database process: Chroma is embedded and
persists to `./chroma_data` next to the backend.

Run tests: `pytest` (from `backend/`).

---

## User flow

1. **Upload your resume** (PDF/TXT/MD) — it's parsed, chunked, embedded into a vector store, and
   its skills are extracted.
2. **Upload one or more job descriptions** — same pipeline, each gets a friendly label ("Job #1",
   "Job #2", ...).
3. **Chat unlocks** once a resume and at least one job are processed. Ask things like:
   - "What skills am I missing for Job #2?"
   - "How does my experience align with this role?"
   - "Help me prepare for an interview for Job #1"

Follow-up questions work naturally without repeating context (see "Conversation memory" below).

---

## Architecture

```
Browser (session_id in localStorage, sent as X-Session-Id header)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Backend                       │
│  POST /resume     POST /jobs     POST /chat   GET /health │
└───────────────┬─────────────────────────────────────────┘
                │
                ▼
   Per-session processing: Parse → Chunk → Embed → Chroma collection
   "career_<session_id>"  +  LLM skill extraction → in-memory SessionState
                │
                ▼
   /chat: resolve active job → rewrite query using recent turns →
   retrieve chunks (vector + BM25 hybrid) → deterministic skill diff →
   build prompt (history + facts + context) → LLM → answer + citations
```

Backend layout (`backend/app/`): `main.py` (app + global error handling), `api.py` (routes),
`session.py` (per-session isolation), `config.py`, `ingestion.py`, `llm.py` (OpenAI wrappers),
`gap_analysis.py` (pure/deterministic), `retrieval.py` (hybrid search), `conversation.py`
(follow-up handling), `chat.py` (prompt composition + guardrails).

---

## Why this isn't just "chat with your docs"

The task's example queries ("what skills am I missing?") need **verifiable comparison**, not just
semantically-similar text. Letting an LLM eyeball a skill gap is a hallucination risk on exactly
the question most likely to be scrutinized. So the system is a hybrid:

1. **Structured extraction**: one LLM call per document (`llm.extract_profile`) reads the *entire*
   document — explicitly instructed not to stop at a "Skills:" list, since resumes commonly
   mention tools only inside experience/project bullets (verified: extraction went from 9 skills
   found via a naive prompt to 16-54 depending on the document, once the prompt explicitly asked
   for a full end-to-end read) — and returns both a flat `skills: []` list and a comprehensive
   narrative `summary` via a strict JSON schema, in a single call (not two, to keep cost/latency
   flat). This also makes the pipeline noticeably more robust to messy PDF text extraction (some
   resume templates produce letter-spaced text like `T e c h L e a d` from pypdf) — the model
   only has to correctly parse that once, here, rather than repeatedly reasoning over raw noisy
   text on every chat turn. Skill embeddings are also computed once here and cached on the
   document, for step 2b below — no extra API calls later.
2. **Deterministic comparison** (`gap_analysis.py`), in two passes, both still fully
   code-driven — no LLM judgment call:
   - **2a. String/alias match** (primary, drives `fit_score`): plain Python set comparison
     with a small synonym map (`JS` → `JavaScript`) and parenthetical-abbreviation handling
     (`"Large Language Models (LLMs)"` matches a plain `"LLMs"` mention).
   - **2b. Semantic fallback** (`add_semantic_matches`): for skills that don't string/alias-match
     anything, compare cached embeddings between the leftover job and resume skills; anything
     above a similarity threshold is surfaced as a separate **"possibly related"** bucket (e.g.
     resume "Vector Databases" vs. job "Pinecone") — shown to the user/LLM as a similarity-based
     suggestion to verify, never silently folded into "matched," so `fit_score` stays conservative
     and defensible.
   - Two scores are reported, not one: **`fit_score`** (confident matches only — the honest,
     defensible number) and **`weighted_fit_score`** (same denominator, but "possibly related"
     pairs count as partial credit, via a plain constant `POSSIBLY_RELATED_WEIGHT = 0.5`, not an
     LLM guess). This avoids the score looking artificially low just because a resume phrases an
     equivalent skill differently (e.g. "AI Engineering" vs. a job requirement listed as plain
     "AI") — the chat prompt explicitly instructs the LLM to present both numbers and explain
     the difference rather than picking one.
3. **Full-context narration for holistic questions**: open-ended questions ("how does my
   experience align?", "should I apply?") are answered by giving the LLM the pre-generated
   **resume/job summaries** (from step 1) plus the **full raw text** of the resume and relevant
   job description(s) — not just top-k retrieved chunks — plus the deterministic skill-diff
   facts, and asking for an explicit qualitative alignment judgment grounded in all of that.
   Retrieval-based chunk similarity is the wrong tool for this: there's no meaningful "search
   query" for an overall comparison, and truncating to the most similar-looking snippets can
   drop exactly the seniority/experience-depth details that matter most — which is exactly what
   the summaries are there to capture. Resumes/JDs are short enough that sending full text
   (capped per document) is cheap and removes this failure mode entirely. Hybrid retrieval is
   still used for the scope-lock guardrail check and for citation snippets in the API response,
   just not for building the LLM's actual reasoning context.

This split is the main engineering decision to defend: the LLM explains verified facts and
grounded text; it never decides the facts themselves.

---

## RAG / LLM approach & decisions

| Area | Choice | Why |
|---|---|---|
| Chunking | `RecursiveCharacterTextSplitter`, 500 chars, 50 overlap | Resumes/JDs are short; small chunks keep retrieval precise |
| Embeddings | OpenAI `text-embedding-3-small` | Good quality/cost trade-off for short documents |
| Vector DB | Chroma, embedded (`PersistentClient`), one collection per session | No server/Docker to manage; per-session collections make cross-user data leakage structurally impossible, not just filtered-away |
| Retrieval | Vector top-k, re-ranked via BM25 + reciprocal rank fusion (`retrieval.py`) — used for the scope-lock guardrail and citation snippets, not for building the LLM's main context (see "Full-context narration" above) | Resume/JD text has exact terminology (tool names, certs) that benefits from keyword matching alongside semantic search; but chunk similarity is the wrong tool for holistic fit questions, so it's scoped to what it's actually good at |
| LLM | `gpt-4o-mini` for chat + extraction | Cost-effective; `gpt-4o` is a documented upgrade path if extraction/answer quality needs to improve |
| Extraction | One combined structured-output call per document returns both a full-document summary and a complete skill list (`llm.extract_profile`) — not two separate calls, and not free text | Reliable/typed skills for deterministic comparison, plus a clean narrative summary (seniority, years of experience, domains) that a flat skill list can't capture, without doubling extraction cost/latency |
| Prompt/context management | System prompt + `<conversation_history>`, `<skill_gap_facts>`, `<context>` tags | Clear separation between instructions, untrusted document data, and verified facts |
| Orchestration | LangChain (`RecursiveCharacterTextSplitter`) for chunking; plain OpenAI SDK calls for embeddings/chat/extraction | Used the framework where it saved real code (splitting), skipped it where a direct SDK call is simpler and more transparent than a chain abstraction |

### Conversation memory (for a natural demo)

A fixed fit for a realistic 5-10 turn demo, without extra LLM calls or infra:

- `SessionState.history`: last 6 turns (3 exchanges), included in the prompt for continuity.
- `active_job_id`: set whenever a question names a job ("Job #2"), reused for follow-ups that
  don't — so "how do I address that gap?" works right after a skill-gap question.
- **Query rewriting for retrieval**: short/ambiguous follow-ups get the previous user turn
  concatenated before embedding, so vector/BM25 search doesn't fail on underspecified text.

No rolling summarization or long-term memory — documented as a deliberate scope cut, not an
oversight (see "What's cut" below).

### Guardrails & security

Enforced at more than the prompt layer, since prompt-only rules can be bypassed:

1. **Scope lock**: system prompt restricts answers to resume/JD content; a retrieval-similarity
   threshold (`MIN_RETRIEVAL_SCORE`) short-circuits to a fixed refusal *before* calling the LLM if
   nothing relevant was retrieved — also saves an API call on clearly off-topic questions.
2. **No action-taking**: the assistant has no tools, function-calling, or filesystem/network
   access — an architectural guarantee, not just an instruction. Retrieved document text and
   conversation history are wrapped in `<context>`/`<conversation_history>` tags with an explicit
   instruction to treat their contents as inert data, never as commands — defends against prompt
   injection embedded inside an uploaded resume/JD.
3. **No internal-details disclosure**: the system prompt forbids revealing itself, the model name,
   config, or other sessions' data. More importantly, this is structural: nothing outside the
   current session's own resume/JD/skill-diff content, and no secrets/config, are ever placed in
   the LLM's context window — so there's nothing to leak even under a successful injection
   attempt. API keys live only in backend env vars, never in prompts, responses, or logs.

Other baseline guardrails: file-type/size validation on uploads, a max of 5 job descriptions per
session (checked before any parsing/LLM call, to fail fast and avoid unbounded per-session cost),
question-length limits, retry once then fail clearly on extraction schema-validation failure, and
a global FastAPI exception handler that converts any OpenAI API error into a clean, generic
message — the raw upstream error (which can include request details) is logged server-side only,
never returned to the client.

### Multi-user isolation

No login system was in scope, so isolation uses an anonymous `session_id`: the frontend generates
a UUID (`crypto.randomUUID()`) on first load, stores it in `localStorage`, and sends it as
`X-Session-Id` on every request. The backend uses it as a hard tenancy boundary — **a separate
Chroma collection per session**, plus a per-session in-memory dict for resume/job/skill state and
conversation history. Using a collection per session (instead of one shared collection with a
metadata filter) means a missed/buggy filter can't leak another user's data — the isolation is
structural. Concurrent users' requests touch entirely independent state, so there's nothing to
race on.

Known limitation: session state is in-memory, so restarting the server clears all sessions. Noted
explicitly below as the production upgrade path.

### Observability

Structured log lines per request (`api.py`): document ingestion latency, chunk/skill counts,
chat latency, resolved active job, and whether a request was refused by the scope-lock guardrail.
Enough to demonstrate the concept for a 1-day build; see "Productionizing" for what a real
deployment would add.

---

## What's cut for this build (and why)

- **No SQLite/Postgres** — session state (resume/job/skills/history) is in-memory only; simplest
  correct option for a single-process, single-day build. Lost on restart — noted, not hidden.
- **No full resume schema** (experience/education/certifications) — only `skills` are extracted
  and used for deterministic comparison; experience-alignment questions go through the RAG path
  instead of a second structured comparison. A full schema is the natural next step.
- **No dedicated `/gap-analysis` or interview-prep endpoints** — folded into the single `/chat`
  endpoint via prompt composition rather than separate routes/UI, to keep the surface area small.
- **BM25 hybrid retrieval is implemented but simple** — in-process `rank_bm25` + RRF, no separate
  search service; fine at this document scale, would need reconsideration at larger scale.
- **No auth/login, no session expiry job** — anonymous session isolation only.
- **Semantic-match threshold (0.75) is a starting estimate, not empirically tuned** — would
  want to calibrate against real resume/JD pairs with more time; also only compares single
  skill phrases, not the fuller context around them.
- **Minimal automated test coverage** — prioritized `gap_analysis.py` (the highest
  hallucination-risk component, and the easiest to test without live API calls) and basic
  ingestion/chunking tests. No LLM-in-the-loop end-to-end tests (would need a live API key and
  cost real tokens per CI run) or frontend tests.
- **No rolling-summary/long-term conversation memory** — fixed 6-turn window only.

---

## Productionizing (hyperscaler deployment)

- **Session/entity state**: move from in-memory to Redis (with TTL-based expiry) or Postgres, so
  state survives restarts and can be shared across multiple backend instances.
- **Vector DB**: embedded Chroma is single-process; for multi-instance scaling, move to a managed
  vector DB (Qdrant Cloud, Pinecone, or Chroma's own hosted offering) reachable by all instances.
- **Ingestion**: move parsing/chunking/embedding/extraction off the request path into an async
  queue (e.g. SQS + worker, or Celery) so uploads don't block on LLM latency, with a
  polling/webhook status endpoint for the frontend.
- **Compute**: containerize the FastAPI app and deploy on a managed container platform (Cloud Run,
  ECS Fargate, or Azure Container Apps) behind a load balancer, autoscaled on request volume.
- **Frontend**: serve as a static build from a CDN (S3+CloudFront, Cloudflare Pages) instead of
  FastAPI's `StaticFiles`, talking to the API over HTTPS with proper CORS locked to the real
  origin (currently wide open for local dev).
- **Auth & PII**: resumes are sensitive personal data. A production version needs real
  authentication (replacing the anonymous session model), encryption at rest for uploaded
  documents, an explicit data retention/deletion policy, and audit logging.
- **Secrets**: OpenAI key via a secrets manager (AWS Secrets Manager / GCP Secret Manager),
  never in plain env files.
- **Rate limiting**: per-session/per-IP limits on uploads and chat to control cost and abuse.
- **Observability**: replace ad-hoc structured logs with a proper LLM observability tool
  (Langfuse or Helicone) for trace-level visibility into retrieval quality, prompt/response pairs,
  token cost, and latency breakdowns; standard metrics/dashboards (Prometheus/Grafana or the
  hyperscaler's native monitoring) for the API layer itself.

---

## Key technical decisions (quick reference)

- **FastAPI** over a heavier framework — small surface area, good async support, easy to reason
  about for a 1-day build.
- **Chroma over Qdrant** — originally planned Qdrant, switched to Chroma once "no Docker, minimal
  ops" became the explicit priority: Chroma runs embedded in-process, so there's no separate
  server to install/manage at all.
- **Deterministic gap analysis instead of an LLM-judged comparison** — the single highest-value
  decision for trustworthiness; explained in detail above.
- **Per-session Chroma collections instead of a shared collection + metadata filter** — trades a
  small amount of resource overhead for structural (not just logical) tenant isolation.
- **Fixed-window conversation memory instead of summarization** — right-sized for a short demo
  conversation without adding LLM calls or moving parts.

---

## How AI tools were used in development

This project was built with AI coding assistance (Devin CLI) in a planning-first workflow: the
architecture, trade-offs, and scope decisions in `PLAN.md` were worked out interactively and
revised several times (initial stack choices, then Qdrant→Chroma, then scope-cut to a 1-day
build, then adding conversation memory and security guardrails) before any code was written. Code
generation followed the plan module-by-module rather than being generated wholesale, and each
piece was run and verified (unit tests, live server smoke tests, error-path testing with a
deliberately invalid API key) rather than accepted on faith. The reasoning behind each decision
above — why Chroma, why deterministic gap analysis, why per-session collections, why a fixed
conversation window — reflects actual trade-off thinking, not just a model's default suggestion;
several of those decisions (switching vector DBs, cutting scope to fit a day, adding the
security section) were explicit course-corrections requested during the process, not the first
draft.

**Do**: use AI assistance to move fast on well-understood, boilerplate-heavy code (endpoint
wiring, test scaffolding, config plumbing) and to get a second opinion on trade-offs before
committing to a design.
**Don't**: accept an architecture or a "what we cut" list without personally verifying it makes
sense for the actual constraints (time budget, real security requirements) — several parts of this
plan were revised specifically because a first pass didn't match the real constraint (e.g. no
Docker available, 1-day deadline, multi-user isolation requirement).

---

## What I'd do differently with more time

- Full resume/JD structured schema (experience, education, seniority) and a real structured
  gap-analysis endpoint/UI instead of folding everything into chat.
- Persistent session store (Postgres/Redis) so a demo can survive a server restart.
- Empirically tune the semantic-match threshold against a labeled resume/JD dataset instead of
  an estimated constant.
- A small LLM-in-the-loop eval harness (golden resume/JD pairs with expected skill gaps) to catch
  regressions in extraction or retrieval quality.
- Real authentication instead of anonymous sessions, given resumes are PII.
- A proper query-rewrite step (small LLM call) for retrieval instead of simple concatenation, if
  conversations needed to go beyond simple follow-ups.

---

## Screenshots

_Add screenshots/demo video here after a live run with a real OpenAI API key._
