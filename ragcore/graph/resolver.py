"""EntityResolver: decides whether a newly extracted entity is the same
real-world thing as one already in the GraphStore.

Kept separate from EntityExtractor (ragcore/graph/base.py) because "extract
candidates from text" and "decide if two candidates are the same real-world
thing" are different concerns with different failure modes. Content-
addressed entity ids (see ragcore/graph/pipeline.py) already collapse exact
canonical-name matches for free — this resolver only needs to fire when a
candidate's normalized name doesn't already match an existing entity, which
mostly happens across books/sources rather than within one.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

from ragcore.graph.types import Entity

if TYPE_CHECKING:
    # Deferred: ragcore.embedding.base imports ragcore.types (for Vector),
    # which would reach back into a not-yet-finished ragcore.types import —
    # see ragcore/graph/pipeline.py for the full story. Embedder is only
    # ever used here as a type hint; resolve() duck-types self._embedder.
    from ragcore.embedding.base import Embedder


@runtime_checkable
class EntityResolver(Protocol):
    def resolve(self, candidate: Entity, existing: Sequence[Entity]) -> str | None:
        """Return the entity_id of an existing entity `candidate` should be
        merged into, or None if it's a genuinely new, distinct entity."""
        ...


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _entity_text(entity: Entity) -> str:
    aliases = entity.extra.get("aliases") or []
    alias_part = f" (aka {', '.join(aliases)})" if aliases else ""
    return f"{entity.entity_type}: {entity.name}{alias_part}"


class EmbeddingEntityResolver:
    """Default EntityResolver: embeds candidate + existing entities of the
    same type in one batched call (reusing the Corpus's own Embedder — no
    separate model or dependency), cosine-compares, and merges above
    `threshold`. Only entities of the same `entity_type` are ever compared.
    """

    def __init__(self, embedder: Embedder, *, threshold: float = 0.87) -> None:
        self._embedder = embedder
        self._threshold = threshold

    def resolve(self, candidate: Entity, existing: Sequence[Entity]) -> str | None:
        same_type = [e for e in existing if e.entity_type == candidate.entity_type]
        if not same_type:
            return None

        texts = [_entity_text(candidate)] + [_entity_text(e) for e in same_type]
        groups = [[text] for text in texts]  # each embedded independently, one request
        vectors = [g[0] for g in self._embedder.embed_groups(groups, input_type="document")]
        candidate_vec, *existing_vecs = vectors

        best_score, best_id = -1.0, None
        for entity, vec in zip(same_type, existing_vecs):
            score = _cosine(candidate_vec, vec)
            if score > best_score:
                best_score, best_id = score, entity.entity_id

        return best_id if best_score >= self._threshold else None
