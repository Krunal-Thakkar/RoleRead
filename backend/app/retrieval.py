"""Retrieval over a session's Chroma collection.

Baseline: plain vector top-k. Hybrid BM25 fusion is included since it was in
scope, but kept simple (in-process, no extra service) and safely degrades to
vector-only if the collection is too small for BM25 to matter.
"""
from dataclasses import dataclass
from typing import List, Optional

from rank_bm25 import BM25Okapi

from .config import settings
from .llm import embed_texts


@dataclass
class RetrievedChunk:
    text: str
    doc_id: str
    doc_type: str
    filename: str
    label: str
    score: float


def _vector_search(collection, query: str, n_results: int, where: Optional[dict]) -> List[RetrievedChunk]:
    query_embedding = embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
    )
    chunks: List[RetrievedChunk] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, distances):
        # Chroma default distance is squared L2 on normalized vectors; convert to a rough similarity in [0,1].
        similarity = max(0.0, 1.0 - dist / 2.0)
        chunks.append(
            RetrievedChunk(
                text=doc,
                doc_id=meta.get("doc_id", ""),
                doc_type=meta.get("doc_type", ""),
                filename=meta.get("filename", ""),
                label=meta.get("label", ""),
                score=similarity,
            )
        )
    return chunks


def _bm25_rerank(query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    if len(chunks) < 2:
        return chunks
    tokenized_corpus = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())

    # Reciprocal rank fusion between vector rank and BM25 rank.
    vector_rank = {id(c): rank for rank, c in enumerate(chunks)}
    bm25_rank = {id(c): rank for rank, c in enumerate(sorted(chunks, key=lambda c: -bm25_scores[chunks.index(c)]))}

    def rrf_score(c) -> float:
        k = 60
        return 1.0 / (k + vector_rank[id(c)]) + 1.0 / (k + bm25_rank[id(c)])

    return sorted(chunks, key=rrf_score, reverse=True)


def retrieve(
    collection,
    query: str,
    doc_ids: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    hybrid: bool = True,
) -> List[RetrievedChunk]:
    n = top_k or settings.top_k
    where = {"doc_id": {"$in": doc_ids}} if doc_ids else None

    # Over-fetch a bit so BM25 re-ranking has something to work with.
    fetch_n = max(n * 3, n)
    chunks = _vector_search(collection, query, fetch_n, where)
    if hybrid and chunks:
        chunks = _bm25_rerank(query, chunks)
    return chunks[:n]
