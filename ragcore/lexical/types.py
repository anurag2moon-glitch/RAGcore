"""Data types for the lexical (keyword/BM25) retrieval channel."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LexicalHit:
    """One keyword-search result. `rank` (1-based, best first) is the only
    thing reciprocal rank fusion (ragcore.retrieval.hybrid.HybridRetriever)
    consumes — a lexical backend's own relevance score is backend-specific
    (SQLite FTS5's bm25() is a small negative number, lower-is-better) and
    never compared directly against a vector store's cosine-derived score."""

    chunk_id: str
    rank: int
