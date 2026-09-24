"""Shared scoring helper: the one score scale every retrieval layer in this
package agrees on.

`LanceDBStore.query()` scores a hit as `1.0 / (1.0 + max(distance, 0.0))`
with `distance` a cosine distance. Every add-on layer that has to invent a
score for something the base vector store didn't itself rank (a
concept-linked chunk, a community/hierarchy summary, a lexical-only hybrid
hit) reproduces this exact transform rather than an arbitrary constant —
that's what lets GraphRetriever's decayed expansion scores, ConceptRetriever's
and HierarchyRetriever's cosine-derived scores, and HybridRetriever's fused
hits all sort correctly against each other and against plain vector hits in
one merged list, and what makes MMRReranker's min-max normalization mean the
same thing regardless of which layer a candidate came from. Getting this
wrong (e.g. emitting a raw reciprocal-rank-fusion score, ~1/60, alongside
these ~0.33-1.0 scores) silently inverts result ordering — see
ragcore/retrieval/hybrid.py's module docstring for the concrete failure mode
this was designed to avoid.
"""

from __future__ import annotations

from typing import Sequence

from ragcore.graph.resolver import _cosine


def cosine_score(query_vector: Sequence[float], vector: Sequence[float] | None) -> float:
    if vector is None:
        return 0.0
    distance = 1.0 - _cosine(query_vector, vector)
    return 1.0 / (1.0 + max(distance, 0.0))
