"""Retrieval quality metrics: Recall@k, Precision@k, nDCG@k, MRR.

Plain functions over (ranked chunk ids, relevant chunk id set) — no
dependency on ragcore types, so they're equally usable against the
7-chapter Pride and Prejudice fixture and the full synthetic library (the
"re-run the eval harness at larger scale" requirement needs the exact same
metric code at both scales for the comparison to mean anything).

There was no eval harness anywhere in ragcore before this suite — no
Recall@k/nDCG/MRR code existed in the library, the tests, or the design
doc. This is a new, permanent addition living under tests/, not shipped as
part of the ragcore package (see docs/INTEGRATION_TEST_FINDINGS.md for why:
metrics are a testing concern, not a library API surface).
"""

from __future__ import annotations

import math
from typing import Sequence


def recall_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of `relevant` that appear in the top k of `ranked_ids`.
    0.0 if `relevant` is empty is undefined territory — callers should
    filter those out; this returns 0.0 rather than raising, since a
    malformed query set is a test-authoring bug, not a runtime one."""
    if not relevant:
        return 0.0
    top_k = set(ranked_ids[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / len(top_k)


def ndcg_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k: relevant chunks have gain 1, everything
    else gain 0 (there is no graded relevance signal in this suite's
    ground truth — a chunk either contains the needle/is the intended
    answer, or it doesn't)."""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 2) for i, cid in enumerate(ranked_ids[:k]) if cid in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked_ids: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant hit in `ranked_ids`, 0.0 if
    none is present anywhere in the ranking."""
    for i, cid in enumerate(ranked_ids):
        if cid in relevant:
            return 1.0 / (i + 1)
    return 0.0
