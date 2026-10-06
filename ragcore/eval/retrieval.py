"""Resolves GoldQuestion.gold_passages against a live Corpus into concrete
chunk_id sets, then scores retrieval quality against them.

Resolution happens at eval-run time, not authoring time (except via the
find_gold_passage authoring helper in ragcore.eval.dataset), so a
re-chunked corpus updates the relevant set automatically rather than a
gold file silently going stale.
"""

from __future__ import annotations

import time
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Sequence

from ragcore.eval.metrics import hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from ragcore.eval.types import GoldPassage, GoldQuestion, RetrievalMetrics

if TYPE_CHECKING:
    from ragcore.corpus import Corpus


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    """Same nearest-rank percentile convention as
    tests/integration/eval/report.py's identical helper — reimplemented,
    not imported, since package code must never depend on tests/ (see
    ragcore.eval.dataset's module docstring for the same rule applied to
    find_gold_passage)."""
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(round(p * (len(sorted_values) - 1))))
    return sorted_values[idx]


def _source_id_for_path(corpus: Corpus, source_path: str) -> str | None:
    target = Path(source_path).resolve()
    for source in corpus.sources():
        if Path(source.path).resolve() == target:
            return source.source_id
    return None


def resolve_relevant_chunk_ids(corpus: Corpus, gold_passages: Sequence[GoldPassage]) -> set[str]:
    """Expands each GoldPassage into concrete, currently-existing chunk_ids
    at the tightest granularity given — exact chunk_ids, else every chunk
    in a (source, chapter_index), else every chunk in a source — by
    scanning the live corpus. A gold passage that can't be resolved (e.g.
    its source was removed) contributes nothing rather than raising."""
    relevant: set[str] = set()
    for gp in gold_passages:
        if gp.chunk_ids:
            relevant.update(gp.chunk_ids)
            continue

        source_id = gp.source_id
        if source_id is None and gp.source_path is not None:
            source_id = _source_id_for_path(corpus, gp.source_path)
        if source_id is None:
            continue

        chunks = corpus.store.get_by_source(source_id)
        if gp.chapter_index is not None:
            chunks = [c for c in chunks if c.meta.chapter_index == gp.chapter_index]
        relevant.update(c.chunk_id for c in chunks)
    return relevant


def score_question(
    corpus: Corpus, question: GoldQuestion, *, k: int = 10
) -> tuple[list[str], set[str], float]:
    """Returns (retrieved_chunk_ids, relevant_chunk_ids, latency_ms) for
    one question."""
    t0 = time.perf_counter()
    results = corpus.retrieve(question.query, k=k)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    retrieved_ids = [r.chunk.chunk_id for r in results]
    relevant = resolve_relevant_chunk_ids(corpus, question.gold_passages)
    return retrieved_ids, relevant, latency_ms


def score_retrieval(
    corpus: Corpus, questions: Sequence[GoldQuestion], *, k: int = 10
) -> tuple[RetrievalMetrics, list[tuple[str, list[str], set[str], float]]]:
    """Aggregate retrieval metrics (mean across questions) plus the raw
    per-question (question_id, retrieved_ids, relevant_ids, latency_ms)
    tuples, so ragcore.eval.report.run_eval can build QuestionResults
    without re-querying the corpus."""
    recalls, precisions, ndcgs, rrs, hits, latencies = [], [], [], [], [], []
    raw: list[tuple[str, list[str], set[str], float]] = []

    usage = getattr(corpus.embedder, "usage", None)
    tokens_before = usage.total_tokens if usage is not None else None

    for q in questions:
        retrieved_ids, relevant, latency_ms = score_question(corpus, q, k=k)
        recalls.append(recall_at_k(retrieved_ids, relevant, k))
        precisions.append(precision_at_k(retrieved_ids, relevant, k))
        ndcgs.append(ndcg_at_k(retrieved_ids, relevant, k))
        rrs.append(reciprocal_rank(retrieved_ids, relevant))
        hits.append(hit_rate_at_k(retrieved_ids, relevant, k))
        latencies.append(latency_ms)
        raw.append((q.question_id, retrieved_ids, relevant, latency_ms))

    total_tokens_used = (usage.total_tokens - tokens_before) if usage is not None else None
    sorted_latencies = sorted(latencies)

    metrics = RetrievalMetrics(
        k=k,
        recall_at_k=fmean(recalls) if recalls else 0.0,
        precision_at_k=fmean(precisions) if precisions else 0.0,
        ndcg_at_k=fmean(ndcgs) if ndcgs else 0.0,
        mrr=fmean(rrs) if rrs else 0.0,
        hit_rate_at_k=fmean(hits) if hits else 0.0,
        latency_p50_ms=_percentile(sorted_latencies, 0.50),
        latency_p95_ms=_percentile(sorted_latencies, 0.95),
        latency_p99_ms=_percentile(sorted_latencies, 0.99),
        total_tokens_used=total_tokens_used,
    )
    return metrics, raw
