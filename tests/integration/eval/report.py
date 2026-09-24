"""Runs a QuerySpec set against a retriever, aggregates Recall@k/Precision@k/
nDCG@k/MRR plus latency, and renders/serializes the result — used both to
gate quality (test_eval_quality.py) and to gate latency (test_query_load.py).
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

from ragcore.types import Retrieved
from tests.integration.eval.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from tests.integration.eval.queryset import QuerySpec

RetrieveFn = Callable[[str, int], list[Retrieved]]


@dataclass(frozen=True)
class EvalSummary:
    label: str
    n_queries: int
    k: int
    recall_at_k: float
    precision_at_k: float
    ndcg_at_k: float
    mrr: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(round(p * (len(sorted_values) - 1))))
    return sorted_values[idx]


def run_eval(retrieve_fn: RetrieveFn, queries: Sequence[QuerySpec], *, k: int = 10, label: str = "") -> EvalSummary:
    if not queries:
        return EvalSummary(
            label=label, n_queries=0, k=k, recall_at_k=0.0, precision_at_k=0.0,
            ndcg_at_k=0.0, mrr=0.0, latency_p50_ms=0.0, latency_p95_ms=0.0,
            latency_p99_ms=0.0, latency_max_ms=0.0,
        )

    recalls, precisions, ndcgs, mrrs, latencies = [], [], [], [], []
    for q in queries:
        t0 = time.perf_counter()
        results = retrieve_fn(q.query, k)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        ranked_ids = [r.chunk.chunk_id for r in results]
        relevant = set(q.relevant)
        recalls.append(recall_at_k(ranked_ids, relevant, k))
        precisions.append(precision_at_k(ranked_ids, relevant, k))
        ndcgs.append(ndcg_at_k(ranked_ids, relevant, k))
        mrrs.append(mrr(ranked_ids, relevant))

    latencies.sort()
    return EvalSummary(
        label=label,
        n_queries=len(queries),
        k=k,
        recall_at_k=statistics.fmean(recalls),
        precision_at_k=statistics.fmean(precisions),
        ndcg_at_k=statistics.fmean(ndcgs),
        mrr=statistics.fmean(mrrs),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        latency_p99_ms=_percentile(latencies, 0.99),
        latency_max_ms=latencies[-1] if latencies else 0.0,
    )


def render_table(summaries: Sequence[EvalSummary]) -> str:
    headers = ["label", "n", "k", "recall@k", "prec@k", "ndcg@k", "mrr", "p50ms", "p95ms", "p99ms"]
    rows = [
        [
            s.label, str(s.n_queries), str(s.k),
            f"{s.recall_at_k:.3f}", f"{s.precision_at_k:.3f}", f"{s.ndcg_at_k:.3f}", f"{s.mrr:.3f}",
            f"{s.latency_p50_ms:.2f}", f"{s.latency_p95_ms:.2f}", f"{s.latency_p99_ms:.2f}",
        ]
        for s in summaries
    ]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    lines = [
        "  ".join(h.ljust(w) for h, w in zip(headers, widths)),
        "  ".join("-" * w for w in widths),
    ]
    for r in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    return "\n".join(lines)
