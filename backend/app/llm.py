"""Thin wrappers around the OpenAI API: embeddings, chat, and structured extraction."""
import json
from dataclasses import dataclass, field
from typing import List, Optional

from openai import OpenAI

from .config import settings

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    client = get_client()
    resp = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in resp.data]


def chat_completion(messages: List[dict], model: Optional[str] = None, temperature: float = 0.2) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=model or settings.chat_model,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


_PROFILE_SCHEMA = {
    "name": "extracted_profile",
    "schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "A comprehensive 150-300 word summary written after reading the ENTIRE document "
                    "end to end, not just the top section."
                ),
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Complete list of distinct skills, technologies, tools, frameworks, platforms, or "
                    "competencies mentioned anywhere in the document, including within experience/"
                    "project bullet points, not just an explicit skills list."
                ),
            },
        },
        "required": ["summary", "skills"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass
class ExtractedProfile:
    summary: str = ""
    skills: List[str] = field(default_factory=list)


def extract_profile(text: str, doc_type: str) -> ExtractedProfile:
    """Read the full document once and extract both a comprehensive narrative summary and a
    complete skill list, in a single LLM call (one call, not two, to keep cost/latency the same
    as skill-only extraction).

    Generating a clean summary here — rather than only ever reasoning over raw document text on
    each chat turn — has two benefits: it captures things a flat skill list can't (seniority,
    years of experience, notable projects/domains), which matters for holistic fit questions;
    and it makes the pipeline more resilient to messy PDF text extraction (e.g. letter-spaced
    text produced by certain resume templates/fonts), since the model only needs to correctly
    parse the noisy text once, here, rather than repeatedly re-reasoning over it per chat turn.
    """
    client = get_client()

    if doc_type == "resume":
        doc_label = "resume"
        summary_instructions = (
            "Summarize the candidate: seniority/role level, total years of experience, key "
            "domains/industries, notable projects or achievements, and overall technical profile."
        )
    else:
        doc_label = "job description"
        summary_instructions = (
            "Summarize the role: title and seniority level, core responsibilities, must-have vs. "
            "nice-to-have requirements, and team/domain context."
        )

    system = (
        f"Read the ENTIRE {doc_label} below, end to end — every section, not just the top or an "
        "explicit list if one exists. The text may have minor OCR/PDF extraction artifacts (odd "
        "spacing, line breaks mid-word); read past those rather than treating them as content.\n\n"
        f"1. {summary_instructions}\n\n"
        "2. Extract every distinct skill, technology, tool, framework, platform, and key competency "
        "mentioned anywhere in the document (Summary, Experience, Projects, Skills sections, etc.). "
        "Resumes commonly mention additional tools only inside prose — e.g. a bullet like 'built "
        "REST APIs using FastAPI and deployed on AWS with Docker' mentions four skills: REST APIs, "
        "FastAPI, AWS, Docker — extract all of them, not just ones in a dedicated skills list.\n\n"
        "Base both the summary and the skill list only on what is explicitly present in the text. "
        "Do not invent or infer anything not stated. Normalize obvious abbreviations (e.g. 'JS' -> "
        "'JavaScript') but do not add unrelated skills that merely sound related."
    )
    # Treat the document text strictly as data, never as instructions.
    user = f"<document>\n{text[:30000]}\n</document>"

    resp = client.chat.completions.create(
        model=settings.extraction_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        response_format={"type": "json_schema", "json_schema": _PROFILE_SCHEMA},
    )
    content = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
        summary = (data.get("summary") or "").strip()
        skills = [s.strip() for s in data.get("skills", []) if isinstance(s, str) and s.strip()]
        return ExtractedProfile(summary=summary, skills=skills)
    except (json.JSONDecodeError, AttributeError):
        return ExtractedProfile(summary="", skills=[])
