"""Parse -> chunk -> embed -> store pipeline for resumes and job descriptions."""
import io
import uuid
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import settings
from .llm import embed_texts

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


def parse_file(filename: str, content: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if lower.endswith(".txt") or lower.endswith(".md"):
        return content.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type for '{filename}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")


def is_allowed_filename(filename: str) -> bool:
    lower = filename.lower()
    return any(lower.endswith(ext) for ext in ALLOWED_EXTENSIONS)


def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_text(text)
    return [c for c in chunks if c.strip()]


def store_chunks(collection, doc_id: str, doc_type: str, filename: str, label: str, chunks: List[str]) -> int:
    if not chunks:
        return 0
    embeddings = embed_texts(chunks)
    ids = [f"{doc_id}::{i}" for i in range(len(chunks))]
    metadatas = [
        {"doc_id": doc_id, "doc_type": doc_type, "filename": filename, "label": label, "chunk_index": i}
        for i in range(len(chunks))
    ]
    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return len(chunks)


def new_doc_id() -> str:
    return uuid.uuid4().hex[:12]
