"""ConceptResolver: decides whether a newly extracted concept is the same
abstract idea as one already in the ConceptStore.

Mirrors ragcore.graph.resolver.EmbeddingEntityResolver exactly, including its
vector-cache strategy — the same "extract vs. resolve are different
concerns" argument applies here, and content-addressed concept ids already
collapse exact label matches for free (see
ragcore.concepts.claude_extractor), so this only needs to fire across
chapters/books where wording drifts. Concepts have no ragcore.graph.types.Entity.entity_type
analogue to pre-filter candidates by, so every existing concept is a
comparison candidate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

from ragcore.concepts.types import Concept
from ragcore.graph.resolver import _cosine

if TYPE_CHECKING:
    # Deferred for the same reason ragcore.graph.resolver defers it: Embedder
    # is only ever used here as a type hint; resolve() duck-types
    # self._embedder.
    from ragcore.embedding.base import Embedder


@runtime_checkable
class ConceptResolver(Protocol):
    def resolve(self, candidate: Concept, existing: Sequence[Concept]) -> str | None:
        """Return the concept_id of an existing concept `candidate` should
        be merged into, or None if it's a genuinely new, distinct concept."""
        ...


def concept_text(concept: Concept) -> str:
    """Text embedded for similarity comparison and for query-time mapping
    (ragcore.concepts.mapper.EmbeddingConceptMapper) — label plus its
    disambiguating description."""
    return f"{concept.label}: {concept.description}"


class EmbeddingConceptResolver:
    """Default ConceptResolver: embeds candidate + existing concepts in one
    batched call (reusing the Corpus's own Embedder — no separate model or
    dependency), cosine-compares, and merges above `threshold`.

    Caches an existing concept's embedding vector, keyed by
    `(concept_id, concept_text(concept))` — see
    ragcore.graph.resolver.EmbeddingEntityResolver's docstring for the full
    rationale, which applies here unchanged: a cache entry is naturally
    invalidated the moment a concept's description changes, and misses are
    still batched into one request alongside the candidate.
    """

    def __init__(self, embedder: Embedder, *, threshold: float = 0.87) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._vector_cache: dict[tuple[str, str], Sequence[float]] = {}

    def resolve(self, candidate: Concept, existing: Sequence[Concept]) -> str | None:
        if not existing:
            return None

        texts_to_embed = [concept_text(candidate)]
        miss_indices: list[int] = []
        existing_vecs: list[Sequence[float] | None] = [None] * len(existing)
        for i, c in enumerate(existing):
            text = concept_text(c)
            cached = self._vector_cache.get((c.concept_id, text))
            if cached is not None:
                existing_vecs[i] = cached
            else:
                miss_indices.append(i)
                texts_to_embed.append(text)

        groups = [[text] for text in texts_to_embed]  # each embedded independently, one request
        embedded = [g[0] for g in self._embedder.embed_groups(groups, input_type="document")]
        candidate_vec = embedded[0]
        for offset, i in enumerate(miss_indices):
            vec = embedded[1 + offset]
            existing_vecs[i] = vec
            self._vector_cache[(existing[i].concept_id, concept_text(existing[i]))] = vec

        best_score, best_id = -1.0, None
        for concept, vec in zip(existing, existing_vecs, strict=True):
            score = _cosine(candidate_vec, vec)
            if score > best_score:
                best_score, best_id = score, concept.concept_id

        return best_id if best_score >= self._threshold else None
