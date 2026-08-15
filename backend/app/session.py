"""Per-session state and isolation.

Every browser gets an anonymous session_id (UUID) which is used as the sole
tenancy boundary: each session gets its own Chroma collection and its own
in-memory entity/conversation state. There is no shared mutable state between
sessions, so concurrent users cannot see or affect each other's data.
"""
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import chromadb
from fastapi import Header, HTTPException

from .config import settings

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9\-]{8,64}$")
_JOB_MENTION_RE = re.compile(r"job\s*#?\s*(\d+)", re.IGNORECASE)
_MULTI_JOB_RE = re.compile(
    r"\bother\s+\d+\b|\bother\s+(two|three|jobs)\b|\ball\s+(jobs|three|the\s+jobs)\b|"
    r"\beach\s+job\b|\bevery\s+job\b|\bboth\s+jobs\b|\bcompare\b|\ball\s+of\s+the\s+jobs\b|"
    r"\bwhich\s+job\b|\bwhich\s+jobs\b|\bwhich\s+one\b|\bwhich\s+role\b|\bwhich\s+position\b|"
    r"\bbest\s+fit\b|\bbest\s+suit(?:s|ed)?\b|\bbest\s+match\b|\bmost\s+suitable\b|"
    r"\bbetter\s+fit\b|\bbetter\s+suited\b|\brank(?:ed|ing)?\s+(?:the\s+)?jobs\b",
    re.IGNORECASE,
)

_chroma_client = chromadb.PersistentClient(
    path=settings.chroma_persist_dir,
    settings=chromadb.config.Settings(anonymized_telemetry=False),
)
_client_lock = threading.Lock()


@dataclass
class DocInfo:
    doc_id: str
    filename: str
    doc_type: str  # "resume" | "job"
    label: str
    skills: List[str] = field(default_factory=list)
    raw_text: str = ""
    # LLM-generated summary from reading the full document once at upload time (see
    # llm.extract_profile) — captures seniority/experience-depth/domain context that a flat
    # skill list can't, and is more robust to noisy PDF text than re-reasoning over raw text.
    summary: str = ""
    # skill -> embedding, computed once at upload time so the semantic-match fallback
    # in gap_analysis.py costs zero extra API calls per chat turn.
    skill_embeddings: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class SessionState:
    session_id: str
    resume: Optional[DocInfo] = None
    jobs: Dict[str, DocInfo] = field(default_factory=dict)  # doc_id -> DocInfo
    job_order: List[str] = field(default_factory=list)  # doc_ids in upload order
    history: List[Dict[str, str]] = field(default_factory=list)  # [{role, content}]
    active_job_id: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def collection(self):
        name = f"career_{self.session_id}"
        return _chroma_client.get_or_create_collection(name=name)

    def next_job_label(self) -> str:
        return f"Job #{len(self.job_order) + 1}"

    def add_history_turn(self, question: str, answer: str) -> None:
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        max_messages = settings.max_history_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def resolve_job_scope(self, question: str, explicit_job_id: Optional[str]) -> tuple[str, List[str]]:
        """Resolve which job(s) the question is about.

        Returns (mode, job_ids) where mode is "single" or "all". Priority:
        explicit job_id > explicit "Job #N" mention > "other jobs"/"compare"/etc.
        phrasing (when more than one job exists) > last active job > the only
        job (if just one) > all jobs (if several exist and nothing else matched).
        """
        if explicit_job_id and explicit_job_id in self.jobs:
            self.active_job_id = explicit_job_id
            return "single", [explicit_job_id]

        match = _JOB_MENTION_RE.search(question)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(self.job_order):
                job_id = self.job_order[idx]
                self.active_job_id = job_id
                return "single", [job_id]

        if len(self.job_order) > 1 and _MULTI_JOB_RE.search(question):
            return "all", list(self.job_order)

        if self.active_job_id and self.active_job_id in self.jobs:
            return "single", [self.active_job_id]

        if len(self.job_order) == 1:
            self.active_job_id = self.job_order[0]
            return "single", list(self.job_order)

        if self.job_order:
            return "all", list(self.job_order)

        return "single", []


_sessions: Dict[str, SessionState] = {}
_sessions_lock = threading.Lock()


def get_or_create_session(session_id: str) -> SessionState:
    with _sessions_lock:
        state = _sessions.get(session_id)
        if state is None:
            state = SessionState(session_id=session_id)
            _sessions[session_id] = state
        return state


def session_id_dependency(x_session_id: Optional[str] = Header(default=None)) -> str:
    if not x_session_id or not _SESSION_ID_RE.match(x_session_id):
        raise HTTPException(
            status_code=400,
            detail="Missing or invalid X-Session-Id header. The client must generate a UUID per browser session.",
        )
    return x_session_id


def get_session(session_id: str = Header(default=None, alias="X-Session-Id")) -> SessionState:
    validated = session_id_dependency(session_id)
    return get_or_create_session(validated)
