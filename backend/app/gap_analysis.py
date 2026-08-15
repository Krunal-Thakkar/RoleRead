"""Deterministic skill comparison between a resume and a job description.

Deliberately NOT an LLM call: the "what skills am I missing" question is the
one most likely to be scrutinized for hallucination, so the actual matching
is done with plain, testable code. The LLM's job (in chat.py) is only to
narrate these pre-computed facts, not to decide them.
"""
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set

# Small hardcoded synonym map to reduce trivial false negatives (e.g. "JS" vs "JavaScript").
# A larger/fuzzy or embedding-based synonym match is a documented stretch goal.
_SYNONYMS = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "gcp": "google cloud platform",
    "aws": "amazon web services",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "ci/cd": "continuous integration and continuous deployment",
    "restful api": "rest api",
    "reactjs": "react",
    "nodejs": "node.js",
    "node": "node.js",
    "llm": "large language model",
    "llms": "large language models",
    "rag": "retrieval-augmented generation",
    "genai": "generative ai",
}

_REVERSE_SYNONYMS: Dict[str, Set[str]] = {}
for _key, _value in _SYNONYMS.items():
    _REVERSE_SYNONYMS.setdefault(_value, set()).add(_key)

_PAREN_RE = re.compile(r"\(([^)]+)\)")


def _aliases(skill: str) -> Set[str]:
    """Return the set of normalized strings a skill can be matched under.

    Handles three common cases beyond a plain lowercase compare:
    - hardcoded synonyms/abbreviations (both directions), e.g. "JS" <-> "JavaScript"
    - a spelled-out term with its own abbreviation in parentheses, e.g.
      "Large Language Models (LLMs)" -> also aliased as "large language models" and "llms"
    - matching either the abbreviation or the spelled-out form against the synonym map,
      so "Large Language Models (LLMs)" also matches a plain "LLM" mention via the synonym map
    """
    s = skill.strip().lower()
    aliases = {s}

    match = _PAREN_RE.search(s)
    if match:
        inner = match.group(1).strip()
        outer = _PAREN_RE.sub("", s).strip().rstrip("-").strip()
        if inner:
            aliases.add(inner)
        if outer:
            aliases.add(outer)

    expanded = set(aliases)
    for a in aliases:
        if a in _SYNONYMS:
            expanded.add(_SYNONYMS[a])
        if a in _REVERSE_SYNONYMS:
            expanded |= _REVERSE_SYNONYMS[a]
        # Loose plural/singular match for short abbreviations (e.g. "llm" vs "llms").
        if a.endswith("s") and len(a) > 3:
            expanded.add(a[:-1])
        else:
            expanded.add(a + "s")

    return {a for a in expanded if a}


@dataclass
class PossibleMatch:
    job_skill: str
    resume_skill: str
    similarity: float


@dataclass
class GapAnalysisResult:
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    extra_skills: List[str] = field(default_factory=list)  # resume skills not required by the JD
    fit_score: float = 0.0  # confident string/alias matches only, in [0, 1] — the honest, defensible number
    possibly_related: List[PossibleMatch] = field(default_factory=list)
    # Same denominator as fit_score, but each "possibly related" pair counts as partial credit
    # (see POSSIBLY_RELATED_WEIGHT). Set by add_semantic_matches; equals fit_score until then.
    # Shown *alongside*, never instead of, fit_score — it's an estimate, not a verified number.
    weighted_fit_score: float = 0.0


# How much credit a "possibly related" (semantic-only) match contributes toward
# weighted_fit_score, relative to a confident string/alias match (1.0). Deliberately
# a plain, tunable constant rather than an LLM judgment call.
POSSIBLY_RELATED_WEIGHT = 0.5


def compute_gap_analysis(resume_skills: List[str], job_skills: List[str]) -> GapAnalysisResult:
    if not job_skills:
        return GapAnalysisResult(matched_skills=[], missing_skills=[], extra_skills=list(resume_skills), fit_score=0.0)

    # Aggregate all resume skill aliases into one lookup set, but remember which original
    # resume skill each alias came from so we can compute "extra" (unused) resume skills.
    resume_alias_to_skill: Dict[str, str] = {}
    for rs in resume_skills:
        for alias in _aliases(rs):
            resume_alias_to_skill.setdefault(alias, rs)
    resume_alias_set = set(resume_alias_to_skill)

    matched: List[str] = []
    missing: List[str] = []
    matched_resume_skills: Set[str] = set()

    for js in job_skills:
        job_aliases = _aliases(js)
        overlap = job_aliases & resume_alias_set
        if overlap:
            matched.append(js)
            for alias in overlap:
                matched_resume_skills.add(resume_alias_to_skill[alias])
        else:
            missing.append(js)

    extra = [rs for rs in resume_skills if rs not in matched_resume_skills]

    fit_score = round(len(matched) / len(job_skills), 3) if job_skills else 0.0

    return GapAnalysisResult(
        matched_skills=sorted(matched, key=str.lower),
        missing_skills=sorted(missing, key=str.lower),
        extra_skills=sorted(extra, key=str.lower),
        fit_score=fit_score,
        weighted_fit_score=fit_score,  # no semantic pass has run yet; equals fit_score until add_semantic_matches
    )


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# Pairs that are lexically/superficially similar (so embedding cosine similarity is
# deceptively high, e.g. "Java" vs "JavaScript" measured at ~0.66) but are actually
# distinct skills — surfacing these as "possibly related" would be actively misleading,
# not just imprecise. Not exhaustive; add more as they're found. Checked
# case-insensitively, order-independent. (True synonyms like "Go"/"Golang" belong in
# the _SYNONYMS map above, not here — this list is only for genuinely different skills.)
_FALSE_FRIENDS = {
    frozenset({"java", "javascript"}),
    frozenset({"sql", "nosql"}),
    frozenset({"c", "c++"}),
    frozenset({"c", "c#"}),
    frozenset({"c++", "c#"}),
}


def _is_false_friend(a: str, b: str) -> bool:
    return frozenset({a.strip().lower(), b.strip().lower()}) in _FALSE_FRIENDS


def add_semantic_matches(
    result: GapAnalysisResult,
    job_skill_vectors: Dict[str, List[float]],
    resume_skill_vectors: Dict[str, List[float]],
    threshold: float = 0.60,
) -> GapAnalysisResult:
    """Second pass over a deterministic GapAnalysisResult: for job skills that didn't
    string/alias-match anything (result.missing_skills), check embedding similarity
    against the resume's remaining unmatched skills (result.extra_skills).

    Deliberately pure/vector-in-vector-out (no network calls here) so it stays
    unit-testable with synthetic vectors — the actual embedding calls happen once,
    at upload time, and are cached on the document (see session.DocInfo), so this
    check costs zero extra API calls per chat turn.

    Matches found this way are surfaced as "possibly related" rather than merged
    into matched_skills/missing_skills — the deterministic string/alias match
    remains the sole source of truth for fit_score (unaffected by this pass), and
    the user/LLM can see explicitly that this pairing is a similarity-based guess,
    not a verified match. A separate weighted_fit_score gives partial credit
    (POSSIBLY_RELATED_WEIGHT) for these pairs, so a resume that clearly covers a
    requirement under different wording (e.g. "AI Engineering" vs. a job asking for
    "AI") isn't scored as if the skill were entirely absent — while fit_score stays
    the conservative, confident-only number for anything that needs a hard claim.
    """
    still_missing: List[str] = []
    possibly_related: List[PossibleMatch] = []

    for job_skill in result.missing_skills:
        job_vec = job_skill_vectors.get(job_skill)
        if not job_vec:
            still_missing.append(job_skill)
            continue

        best_sim = 0.0
        best_resume_skill = None
        for resume_skill in result.extra_skills:
            if _is_false_friend(job_skill, resume_skill):
                continue
            resume_vec = resume_skill_vectors.get(resume_skill)
            if not resume_vec:
                continue
            sim = _cosine_similarity(job_vec, resume_vec)
            if sim > best_sim:
                best_sim = sim
                best_resume_skill = resume_skill

        if best_resume_skill is not None and best_sim >= threshold:
            possibly_related.append(
                PossibleMatch(job_skill=job_skill, resume_skill=best_resume_skill, similarity=round(best_sim, 3))
            )
        else:
            still_missing.append(job_skill)

    result.missing_skills = sorted(still_missing, key=str.lower)
    result.possibly_related = sorted(possibly_related, key=lambda m: m.job_skill.lower())

    total_required = len(result.matched_skills) + len(result.missing_skills) + len(result.possibly_related)
    if total_required:
        weighted_count = len(result.matched_skills) + POSSIBLY_RELATED_WEIGHT * len(result.possibly_related)
        result.weighted_fit_score = round(weighted_count / total_required, 3)

    return result
