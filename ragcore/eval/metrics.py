"""Pure retrieval-quality metrics — no ragcore imports at all, deliberately,
so this module is trivially unit-testable and has no coupling to how a
"relevant" set was determined. Reimplements the same formulas
tests/integration/eval/metrics.py already uses for ragcore's own regression
testing (binary relevance, empty-relevant returns 0.0 rather than raising —
a malformed query is an authoring bug, not a runtime one), rather than
importing them: package code must never import from tests/.
"""

from __future__ import annotations

import math
from typing import Sequence


def recall_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(ranked_ids[:k]) & relevant)
    return hits / len(relevant)


def precision_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = len(set(ranked_ids[:k]) & relevant)
    return hits / k


def hit_rate_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return 1.0 if set(ranked_ids[:k]) & relevant else 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 2) for i, chunk_id in enumerate(ranked_ids[:k]) if chunk_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(ranked_ids: Sequence[str], relevant: set[str]) -> float:
    """One query's reciprocal rank. MRR (mean reciprocal rank) is this
    averaged across questions — see ragcore.eval.report.run_eval."""
    if not relevant:
        return 0.0
    for i, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / i
    return 0.0
