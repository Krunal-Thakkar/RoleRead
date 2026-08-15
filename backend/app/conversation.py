"""Incremental conversation handling: active-job resolution and retrieval query rewriting.

Keeps a fixed-window history per session (see SessionState) so a demo can ask
natural follow-up questions ("what about that?", "how do I address it in an
interview?") without repeating which job it's about.
"""
import re
from typing import List, Optional

from .session import SessionState

_SHORT_QUESTION_CHARS = 20
_PRONOUN_FOLLOWUP_RE = re.compile(
    r"\b(that|it|this|those|these|them)\b.*\b(gap|skill|missing|address|role|position|interview)\b"
    r"|\bwhat about\b|\bhow about\b",
    re.IGNORECASE,
)
_HAS_JOB_MENTION_RE = re.compile(r"job\s*#?\s*\d+", re.IGNORECASE)


def rewrite_query_for_retrieval(session: SessionState, question: str) -> str:
    """Concatenate the previous user turn when the current question looks like an
    under-specified pronoun-style follow-up ("what about that?"), so vector/BM25
    retrieval still has enough signal.

    Deliberately conservative: if the question already names a specific job
    ("Job #2") or isn't short/pronoun-shaped, it's treated as self-contained and
    left alone — concatenating an unrelated previous turn (e.g. about a different
    job) would dilute retrieval rather than help it.
    """
    if not session.history:
        return question

    is_short = len(question.strip()) < _SHORT_QUESTION_CHARS
    is_pronoun_followup = bool(_PRONOUN_FOLLOWUP_RE.search(question))
    if not (is_short or is_pronoun_followup):
        return question

    if _HAS_JOB_MENTION_RE.search(question):
        return question

    previous_user_turns = [m["content"] for m in session.history if m["role"] == "user"]
    if not previous_user_turns:
        return question

    return f"{previous_user_turns[-1]} {question}"


def recent_turns_as_messages(session: SessionState, max_turns: Optional[int] = None) -> List[dict]:
    n = (max_turns or 3) * 2
    return session.history[-n:]
