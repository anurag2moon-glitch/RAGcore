"""Data types for the evaluation harness.

Distinct from tests/integration/eval/ (which stays exactly as it is — a
harness for ragcore's *own* regression testing against synthetic-library
fixtures, deliberately not part of the public package per its own
docstring). This is a general-purpose tool for a *consumer's* corpus and
questions: a storable gold-question format, retrieval scoring against it,
and an optional groundedness score reusing ragcore.verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class GoldPassage:
    """One passage a GoldQuestion expects to be retrieved. Matched against a
    retrieved chunk at whichever granularity is available — see
    ragcore.eval.retrieval.matches for the exact tiering — so a gold set
    survives a re-chunked corpus rather than silently going to zero.
    """

    source_id: str | None = None
    source_path: str | None = None
    """Portable across re-ingestion when source_id (a content hash) would
    change — the field to prefer when authoring a gold set by hand."""
    chapter_index: int | None = None
    chunk_ids: tuple[str, ...] = ()
    """Optional, tightest match when known — see
    ragcore.eval.dataset.find_gold_passage for an authoring helper that
    resolves one of these directly from real ingested chunk text."""


@dataclass(frozen=True)
class GoldQuestion:
    question_id: str
    query: str
    gold_passages: tuple[GoldPassage, ...]
    expected_facts: tuple[str, ...] = ()
    """Optional freeform facts the answer should mention — not scored today
    (only groundedness is), reserved for a future answer-correctness
    metric; kept in the stored format now so a gold set doesn't need a
    schema migration to grow into that later."""
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "query": self.query,
            "gold_passages": [
                {
                    "source_id": p.source_id,
                    "source_path": p.source_path,
                    "chapter_index": p.chapter_index,
                    "chunk_ids": list(p.chunk_ids),
                }
                for p in self.gold_passages
            ],
            "expected_facts": list(self.expected_facts),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GoldQuestion:
        return cls(
            question_id=d["question_id"],
            query=d["query"],
            gold_passages=tuple(
                GoldPassage(
                    source_id=p.get("source_id"),
                    source_path=p.get("source_path"),
                    chapter_index=p.get("chapter_index"),
                    chunk_ids=tuple(p.get("chunk_ids") or ()),
                )
                for p in d["gold_passages"]
            ),
            expected_facts=tuple(d.get("expected_facts") or ()),
            notes=d.get("notes"),
        )


@dataclass(frozen=True)
class RetrievalMetrics:
    k: int
    recall_at_k: float
    precision_at_k: float
    ndcg_at_k: float
    mrr: float
    hit_rate_at_k: float

    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    """Wall-clock Corpus.retrieve() latency across the question set — real
    for any run, independent of which Embedder is configured."""

    total_tokens_used: int | None = None
    """Sum of corpus.embedder.usage.total_tokens consumed by this eval
    run's own query embeddings, if the configured Embedder exposes a
    `.usage` (ragcore.embedding.cost.UsageTracker) — None, never a
    guessed number, when it doesn't (e.g. a test double with no real
    cost to track). Measured as a before/after delta around the whole
    run, not attributed per-question: Embedder.embed_query() is one call
    per question in every default stack, but attributing a shared-batch
    embedder's usage to individual questions isn't generally well-defined,
    so this stays a run-level total rather than claiming per-question
    precision it doesn't have."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "mrr": self.mrr,
            "hit_rate_at_k": self.hit_rate_at_k,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "total_tokens_used": self.total_tokens_used,
        }


@dataclass(frozen=True)
class GroundednessMetrics:
    mean_supported_ratio: float
    n_unsupported_claims: int
    n_questions_scored: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_supported_ratio": self.mean_supported_ratio,
            "n_unsupported_claims": self.n_unsupported_claims,
            "n_questions_scored": self.n_questions_scored,
        }


@dataclass(frozen=True)
class QuestionResult:
    question_id: str
    query: str
    retrieved_chunk_ids: tuple[str, ...]
    matched_gold_count: int
    gold_count: int
    reciprocal_rank: float
    latency_ms: float = 0.0
    groundedness: float | None = None
    unsupported_claims: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "query": self.query,
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "matched_gold_count": self.matched_gold_count,
            "gold_count": self.gold_count,
            "reciprocal_rank": self.reciprocal_rank,
            "latency_ms": self.latency_ms,
            "groundedness": self.groundedness,
            "unsupported_claims": list(self.unsupported_claims),
        }


@dataclass(frozen=True)
class EvalReport:
    k: int
    n_questions: int
    retrieval: RetrievalMetrics
    groundedness: GroundednessMetrics | None
    per_question: tuple[QuestionResult, ...]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "n_questions": self.n_questions,
            "retrieval": self.retrieval.to_dict(),
            "groundedness": self.groundedness.to_dict() if self.groundedness else None,
            "per_question": [q.to_dict() for q in self.per_question],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalReport:
        groundedness = d.get("groundedness")
        return cls(
            k=d["k"],
            n_questions=d["n_questions"],
            retrieval=RetrievalMetrics(**d["retrieval"]),
            groundedness=GroundednessMetrics(**groundedness) if groundedness else None,
            per_question=tuple(
                QuestionResult(
                    question_id=q["question_id"],
                    query=q["query"],
                    retrieved_chunk_ids=tuple(q["retrieved_chunk_ids"]),
                    matched_gold_count=q["matched_gold_count"],
                    gold_count=q["gold_count"],
                    reciprocal_rank=q["reciprocal_rank"],
                    latency_ms=q.get("latency_ms", 0.0),
                    groundedness=q.get("groundedness"),
                    unsupported_claims=tuple(q.get("unsupported_claims") or ()),
                )
                for q in d["per_question"]
            ),
            created_at=d.get("created_at", ""),
        )
