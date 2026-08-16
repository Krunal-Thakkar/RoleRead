import logging
import time

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel

from .chat import answer_question
from .config import settings
from .ingestion import chunk_text, is_allowed_filename, new_doc_id, parse_file, store_chunks
from .llm import embed_texts, extract_profile
from .session import DocInfo, SessionState, get_session

router = APIRouter()
logger = logging.getLogger("career_assistant")


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    label: str
    skills: list[str]
    summary: str
    chunk_count: int


class ChatRequest(BaseModel):
    question: str
    job_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    active_job_id: str | None


def _embed_skills(skills: list[str]) -> dict[str, list[float]]:
    """Embed extracted skills once at upload time, so gap_analysis's semantic-match
    fallback (see add_semantic_matches) never needs a fresh API call per chat turn."""
    if not skills:
        return {}
    vectors = embed_texts(skills)
    return dict(zip(skills, vectors))


# Maps the doc_type param used internally ("resume" / "job") to the document_type
# classification values extract_profile's LLM call actually returns ("resume" / "job_description").
_EXPECTED_DOCUMENT_TYPE = {"resume": "resume", "job": "job_description"}
_UPLOAD_LABEL = {"resume": "a resume", "job": "a job description"}


def _reject_if_wrong_document_type(profile, doc_type: str) -> None:
    """Content-based upload validation: reject a file that clearly isn't what it was
    uploaded as (e.g. an invoice uploaded as a "resume"), based on what the extraction
    LLM actually read in the document — not just its filename/extension.

    Deliberately lenient when classification is unavailable (empty `document_type`, e.g. a
    malformed LLM response) — never block an upload on an inconclusive signal, only on a
    confident mismatch.
    """
    if not profile.document_type:
        return
    expected = _EXPECTED_DOCUMENT_TYPE[doc_type]
    if profile.document_type == expected:
        return
    label = _UPLOAD_LABEL[doc_type]
    if profile.document_type == "other":
        detail = (
            f"This file doesn't look like {label}. {profile.summary or 'Please upload the correct document.'}"
        )
    else:
        other_label = _UPLOAD_LABEL["job" if doc_type == "resume" else "resume"]
        detail = f"This file looks like {other_label}, not {label}. Please upload {label} instead."
    raise HTTPException(status_code=400, detail=detail)


async def _read_and_validate(file: UploadFile) -> bytes:
    if not is_allowed_filename(file.filename or ""):
        raise HTTPException(status_code=400, detail="Only .pdf, .txt, and .md files are supported.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=400, detail="File too large (max 10MB).")
    return content


@router.post("/resume", response_model=UploadResponse)
async def upload_resume(file: UploadFile, session: SessionState = Depends(get_session)):
    content = await _read_and_validate(file)
    t0 = time.time()
    text = parse_file(file.filename, content)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from this file.")

    chunks = chunk_text(text)
    doc_id = new_doc_id()
    profile = extract_profile(text, doc_type="resume")
    _reject_if_wrong_document_type(profile, doc_type="resume")
    skill_embeddings = _embed_skills(profile.skills)

    with session.lock:
        store_chunks(session.collection(), doc_id, "resume", file.filename, "resume", chunks)
        session.resume = DocInfo(
            doc_id=doc_id,
            filename=file.filename,
            doc_type="resume",
            label="resume",
            skills=profile.skills,
            raw_text=text,
            summary=profile.summary,
            skill_embeddings=skill_embeddings,
        )

    logger.info(
        "resume_ingested session=%s doc_id=%s chunks=%d skills=%d latency_s=%.2f",
        session.session_id, doc_id, len(chunks), len(profile.skills), time.time() - t0,
    )
    return UploadResponse(
        doc_id=doc_id, filename=file.filename, label="resume", skills=profile.skills,
        summary=profile.summary, chunk_count=len(chunks),
    )


@router.post("/jobs", response_model=UploadResponse)
async def upload_job(file: UploadFile, session: SessionState = Depends(get_session)):
    if len(session.jobs) >= settings.max_jobs_per_session:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum of {settings.max_jobs_per_session} job descriptions per session.",
        )
    content = await _read_and_validate(file)
    t0 = time.time()
    text = parse_file(file.filename, content)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from this file.")

    chunks = chunk_text(text)
    doc_id = new_doc_id()
    profile = extract_profile(text, doc_type="job")
    _reject_if_wrong_document_type(profile, doc_type="job")
    skill_embeddings = _embed_skills(profile.skills)

    with session.lock:
        label = session.next_job_label()
        store_chunks(session.collection(), doc_id, "job", file.filename, label, chunks)
        doc_info = DocInfo(
            doc_id=doc_id,
            filename=file.filename,
            doc_type="job",
            label=label,
            skills=profile.skills,
            raw_text=text,
            summary=profile.summary,
            skill_embeddings=skill_embeddings,
        )
        session.jobs[doc_id] = doc_info
        session.job_order.append(doc_id)
        if session.active_job_id is None:
            session.active_job_id = doc_id

    logger.info(
        "job_ingested session=%s doc_id=%s label=%s chunks=%d skills=%d latency_s=%.2f",
        session.session_id, doc_id, label, len(chunks), len(profile.skills), time.time() - t0,
    )
    return UploadResponse(
        doc_id=doc_id, filename=file.filename, label=label, skills=profile.skills,
        summary=profile.summary, chunk_count=len(chunks),
    )


@router.get("/documents")
async def list_documents(session: SessionState = Depends(get_session)):
    return {
        "resume": (
            {
                "doc_id": session.resume.doc_id,
                "filename": session.resume.filename,
                "skills": session.resume.skills,
                "summary": session.resume.summary,
            }
            if session.resume
            else None
        ),
        "jobs": [
            {
                "doc_id": d,
                "label": session.jobs[d].label,
                "filename": session.jobs[d].filename,
                "skills": session.jobs[d].skills,
                "summary": session.jobs[d].summary,
            }
            for d in session.job_order
        ],
        "active_job_id": session.active_job_id,
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, session: SessionState = Depends(get_session)):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not session.resume:
        raise HTTPException(status_code=400, detail="Upload a resume before asking questions.")
    if not session.jobs:
        raise HTTPException(status_code=400, detail="Upload at least one job description before asking questions.")

    t0 = time.time()
    result = answer_question(session, req.question, req.job_id)
    logger.info(
        "chat session=%s active_job=%s refused=%s latency_s=%.2f",
        session.session_id, result.active_job_id, result.refused, time.time() - t0,
    )
    return ChatResponse(answer=result.answer, sources=result.sources, active_job_id=result.active_job_id)


@router.get("/health")
async def health():
    return {"status": "ok"}
