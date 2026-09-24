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
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _entity_text(entity: Entity) -> str:
    """Text embedded for similarity comparison. Includes known relations
    (`extra["relations"]`, short strings like "danced_with Mr. Bingley" —
    populated by ragcore.graph.pipeline.build_graph from same-batch
    extraction results, and accumulated across merges by GraphStore
    implementations) alongside name/aliases.

    This is a deliberately cheap, partial mitigation for a known gap: an
    entity introduced only by relational description ("the eldest Miss
    Bennet") has no name to lexically match against. Giving the embedder
    shared relational vocabulary (e.g. both entities' text mentioning "Mr.
    Bennet") sometimes recovers the match — but only when the two mentions
    actually share a common, already-linked entity; it is not a general
    fix. See CLAUDE.md's "Known limitation: relationally-described
    entities" for why a more complete fix (cardinality-aware relation
    reasoning) was deliberately not built.
    """
    aliases = entity.extra.get("aliases") or []
    relations = entity.extra.get("relations") or []
    text = f"{entity.entity_type}: {entity.name}"
    if aliases:
        text += f" (aka {', '.join(aliases)})"
    if relations:
        text += f" [{'; '.join(relations)}]"
    return text


class EmbeddingEntityResolver:
    """Default EntityResolver: embeds candidate + existing entities of the
    same type in one batched call (reusing the Corpus's own Embedder — no
    separate model or dependency), cosine-compares, and merges above
    `threshold`. Only entities of the same `entity_type` are ever compared.

    Caches an existing entity's embedding vector, keyed by
    `(entity_id, _entity_text(entity))` — so re-resolving against the same
    (unchanged) existing entity set, which is the common case across many
    candidates in one chapter and many chapters in one build_graph() run,
    costs one embedding call *per distinct existing entity* rather than one
    per *candidate × existing entity*. Keying on the text (not just the id)
    means a cache entry is naturally invalidated the moment that entity's
    aliases/relations change (e.g. after a merge updates `extra`) — the new
    text simply misses and gets embedded fresh, so this can never return a
    stale vector for a since-changed entity. Cache misses are still batched
    into one request alongside the candidate, so the common "first time we
    see this existing entity" case doesn't regress into one API call per
    entity. Unbounded for the lifetime of the resolver instance — fine for
    one build_graph() run's entity population; a resolver reused indefinitely
    across many independent runs would grow it without eviction.
    """

    def __init__(self, embedder: Embedder, *, threshold: float = 0.87) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._vector_cache: dict[tuple[str, str], Sequence[float]] = {}

    def resolve(self, candidate: Entity, existing: Sequence[Entity]) -> str | None:
        same_type = [e for e in existing if e.entity_type == candidate.entity_type]
        if not same_type:
            return None

        texts_to_embed = [_entity_text(candidate)]
        miss_indices: list[int] = []
        existing_vecs: list[Sequence[float] | None] = [None] * len(same_type)
        for i, e in enumerate(same_type):
            text = _entity_text(e)
            cached = self._vector_cache.get((e.entity_id, text))
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
            self._vector_cache[(same_type[i].entity_id, _entity_text(same_type[i]))] = vec

        best_score, best_id = -1.0, None
        for entity, vec in zip(same_type, existing_vecs, strict=True):
            score = _cosine(candidate_vec, vec)
            if score > best_score:
                best_score, best_id = score, entity.entity_id

        return best_id if best_score >= self._threshold else None
