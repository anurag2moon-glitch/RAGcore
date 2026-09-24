"""Test doubles for the integration suite: a semantically-ordered
deterministic embedder, a mock entity extractor driven by planted ground
truth, and thin fault-injection wrappers over real components.

Everything here is zero-network, zero-cost, and deterministic — the whole
point of the suite is exercising real ragcore code paths (real
StructuralChunker, real LanceDBStore, real SqliteGraphStore, real
build_graph, real GraphRetriever) at production scale without an API key or
a bill.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from ragcore.concepts.claude_extractor import _derive_edges, _concept_id
from ragcore.concepts.types import Concept, ConceptEdge
from ragcore.errors import EmbeddingError, StoreError
from ragcore.graph.types import Entity, Relation
from ragcore.stores.base import VectorStore
from ragcore.types import Chunk, Filter, Retrieved, SourceInfo, Vector

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class DeterministicEmbedder:
    """Hashed bag-of-words embedder: deterministic and zero-cost like
    `tests/fakes.py`'s `FakeEmbedder`, but *semantically ordered* — its
    vectors track real lexical overlap, not noise. `FakeEmbedder` hashes
    the whole text into an opaque vector, which is fine for proving the
    pipeline wires together but useless for measuring retrieval quality: two
    chunks about the same thing don't score closer to each other than to a
    random chunk. This embedder does, which is what makes Recall@k/nDCG/MRR
    against planted needle queries (see `synth/library.py`) mean something.

    Each token hashes into one of `dim` buckets via feature hashing, with a
    sublinear (log1p term-frequency) weight; the vector is then
    L2-normalized so cosine similarity approximates a bag-of-words overlap
    score. `embed_groups` blends a small fraction of each group's centroid
    into every member vector before normalizing, so contextualized-group
    behavior (the reason `ChunkGroup`/`embed_groups` exists at all, rather
    than a flat per-chunk `embed`) is actually exercised, not bypassed.
    """

    def __init__(self, dim: int = 256, *, context_blend: float = 0.15) -> None:
        self.name = "deterministic-hashed-bow"
        self.dim = dim
        self._context_blend = context_blend

    def _bow(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokenize(text):
            idx = int(hashlib.sha1(tok.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        for i, v in enumerate(vec):
            if v:
                vec[i] = math.log1p(v)
        return vec

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return vec
        return [x / norm for x in vec]

    def embed_groups(
        self,
        groups: Sequence[Sequence[str]],
        *,
        input_type: Literal["document", "query", None] = "document",
    ) -> list[list[Vector]]:
        out: list[list[Vector]] = []
        for group in groups:
            raw = [self._bow(t) for t in group]
            if len(raw) > 1 and self._context_blend > 0:
                centroid = [sum(v[i] for v in raw) / len(raw) for i in range(self.dim)]
                blend = self._context_blend
                blended = [
                    [(1 - blend) * v[i] + blend * centroid[i] for i in range(self.dim)]
                    for v in raw
                ]
            else:
                blended = raw
            out.append([self._normalize(v) for v in blended])
        return out

    def embed_query(self, text: str) -> Vector:
        return self._normalize(self._bow(text))


@dataclass(frozen=True)
class EntityMention:
    """One entity's appearance in one chapter of one book, as planted by
    `synth/library.py`'s ground truth. `name` is the surface form used in
    *this* chapter — naming drift across chapters/books is expressed by
    giving the same `canonical_id` different `name`s in different mentions,
    the same way independent per-chapter LLM extraction would."""

    canonical_id: str
    """Ground-truth identity this mention refers to — never written to the
    graph directly; only used by MockEntityExtractor to look up what to
    emit, and by tests to check whether the resolver correctly merged (or
    correctly kept separate) the entities that came from it."""
    name: str
    entity_type: str
    aliases: tuple[str, ...] = ()
    relates_to: tuple[tuple[str, str], ...] = ()
    """(relation_type, other canonical_id) pairs also mentioned in this
    chapter — mirrors how ragcore.graph.pipeline.build_graph derives
    relation hints from same-batch extraction."""


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


def _entity_id(entity_type: str, name: str) -> str:
    return f"entity:{entity_type}:{_slug(name)}"


class MockEntityExtractor:
    """Deterministic EntityExtractor double driven by a planted mention
    script (`dict[context_group_id, list[EntityMention]]`), instead of a
    real Claude call. Mirrors `tests/pride_prejudice_fixture.py`'s hand-
    authored-fixture pattern, generalized to run over a whole synthetic
    library: naming drift is expressed by the script giving the same
    `canonical_id` a different surface `name` in different chapters, which
    is exactly how independent per-chapter LLM extraction produces drift in
    the first place (`ClaudeEntityExtractor` calls `extract()` once per
    chapter with no memory of prior chapters' naming choices).

    A chapter with no script entry (most of any generated book — the
    scripts only plant a small set of entities to resolve against) yields
    `([], [])`, matching `FakeEntityExtractor`'s behavior in the existing
    Pride and Prejudice fixture.
    """

    def __init__(self, mentions_by_group: dict[str, list[EntityMention]]) -> None:
        self._mentions_by_group = mentions_by_group
        self.calls: list[list[str]] = []

    def extract(self, chunks: Sequence[Chunk]) -> tuple[list[Entity], list[Relation]]:
        chunk_ids = [c.chunk_id for c in chunks]
        self.calls.append(chunk_ids)
        if not chunks:
            return [], []
        group_id = chunks[0].meta.context_group_id
        mentions = self._mentions_by_group.get(group_id, [])
        if not mentions:
            return [], []

        by_canonical = {m.canonical_id: m for m in mentions}

        def _chunks_mentioning(m: EntityMention) -> tuple[str, ...]:
            # Mirrors the real ClaudeEntityExtractor's chunk_indices: link
            # an entity only to the specific chunk(s) whose text actually
            # contains its name or an alias — not blanket-linked to every
            # chunk of the chapter. Falls back to the whole chapter only if
            # no literal match is found (defensive; shouldn't happen for
            # planted mentions, whose sentences always contain `m.name`
            # verbatim), so a lookup miss degrades rather than drops the
            # entity's provenance entirely.
            needles = (m.name, *m.aliases)
            hits = tuple(c.chunk_id for c in chunks if any(n in c.text for n in needles))
            return hits or tuple(chunk_ids)

        chunk_ids_by_mention = {m.canonical_id: _chunks_mentioning(m) for m in mentions}
        entities = [
            Entity(
                entity_id=_entity_id(m.entity_type, m.name),
                name=m.name,
                entity_type=m.entity_type,
                source_chunk_ids=chunk_ids_by_mention[m.canonical_id],
                extra={"aliases": list(m.aliases)} if m.aliases else {},
            )
            for m in mentions
        ]

        relations: list[Relation] = []
        seen_pairs: set[tuple[str, str, str]] = set()
        for m in mentions:
            src = by_canonical[m.canonical_id]
            for rel_type, other_canonical in m.relates_to:
                tgt = by_canonical.get(other_canonical)
                if tgt is None:
                    continue  # the related entity isn't mentioned in this chapter
                src_id = _entity_id(src.entity_type, src.name)
                tgt_id = _entity_id(tgt.entity_type, tgt.name)
                key = (rel_type, src_id, tgt_id)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                relations.append(
                    Relation(
                        relation_id=f"relation:{rel_type}:{src_id}:{tgt_id}",
                        source_entity_id=src_id,
                        target_entity_id=tgt_id,
                        relation_type=rel_type,
                        source_chunk_ids=tuple(chunk_ids),
                    )
                )

        return entities, relations


@dataclass(frozen=True)
class ConceptMention:
    """One concept's appearance in one chapter, as planted by an integration
    test — the ragcore.concepts analogue of EntityMention above. `marker` is
    a literal substring of real ingested chunk text (never an authoring-time
    offset) used to decide which of the chapter's chunks this concept links
    to, mirroring how the real ClaudeConceptExtractor only cites the
    specific chunk(s) it actually found a concept discussed in."""

    canonical_id: str
    label: str
    description: str
    marker: str


class MockConceptExtractor:
    """Deterministic ConceptExtractor double driven by a planted mention
    script (`dict[context_group_id, list[ConceptMention]]`), instead of a
    real Claude call — mirrors MockEntityExtractor above. Edges are derived
    the same way the real ClaudeConceptExtractor derives them: from which
    planted concepts actually land on the same chunk, via the extractor
    module's own `_derive_edges` (reused, not reimplemented)."""

    model = "mock-concept-extractor"

    def __init__(self, mentions_by_group: dict[str, list[ConceptMention]]) -> None:
        self._mentions_by_group = mentions_by_group
        self.calls: list[list[str]] = []

    def extract(self, chunks: Sequence[Chunk]) -> tuple[list[Concept], list[ConceptEdge]]:
        chunk_ids = [c.chunk_id for c in chunks]
        self.calls.append(chunk_ids)
        if not chunks:
            return [], []
        group_id = chunks[0].meta.context_group_id
        mentions = self._mentions_by_group.get(group_id, [])
        if not mentions:
            return [], []

        def _chunks_mentioning(m: ConceptMention) -> tuple[str, ...]:
            hits = tuple(c.chunk_id for c in chunks if m.marker in c.text)
            return hits or tuple(chunk_ids)

        concepts = [
            Concept(
                concept_id=_concept_id(m.label),
                label=m.label,
                description=m.description,
                source_chunk_ids=_chunks_mentioning(m),
            )
            for m in mentions
        ]
        return concepts, _derive_edges(concepts)


# --------------------------------------------------------------------------
# Fault injection: thin decorators over a REAL component, so everything up
# to the injected failure point runs the real code path.
# --------------------------------------------------------------------------


class FailingEmbedder:
    """Wraps a real Embedder; raises EmbeddingError on the Nth
    `embed_groups` call onward (1-indexed) — simulates a mid-ingestion kill
    (network drop, OOM, process kill) partway through a multi-group book."""

    def __init__(self, inner: Any, *, fail_after_n_calls: int) -> None:
        self._inner = inner
        self.name = inner.name
        self.dim = inner.dim
        self._fail_after = fail_after_n_calls
        self._calls = 0
        self.lock = threading.Lock()

    def embed_groups(
        self,
        groups: Sequence[Sequence[str]],
        *,
        input_type: Literal["document", "query", None] = "document",
    ) -> list[list[Vector]]:
        with self.lock:
            self._calls += 1
            call_n = self._calls
        if call_n >= self._fail_after:
            raise EmbeddingError(
                f"simulated embedding failure on call {call_n} (fail_after_n_calls={self._fail_after})"
            )
        return self._inner.embed_groups(groups, input_type=input_type)

    def embed_query(self, text: str) -> Vector:
        return self._inner.embed_query(text)


class FailingStore:
    """Wraps a real VectorStore; raises on `fail_on` ("upsert" or "delete")
    the `fail_after_n_calls`'th time that method is invoked (1-indexed) —
    simulates disk-full (ENOSPC) or a killed process mid-write. Every other
    method passes straight through to the real store, so reads against
    whatever was already committed still work exactly as they would against
    the real backend.
    """

    def __init__(
        self,
        inner: VectorStore,
        *,
        fail_on: Literal["upsert", "delete"],
        fail_after_n_calls: int = 1,
        exc: Exception | None = None,
    ) -> None:
        self._inner = inner
        self._fail_on = fail_on
        self._fail_after = fail_after_n_calls
        self._calls = 0
        self._exc = exc or StoreError("simulated disk-full (ENOSPC) during write")

    def _maybe_fail(self, method: str) -> None:
        if method != self._fail_on:
            return
        self._calls += 1
        if self._calls >= self._fail_after:
            raise self._exc

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        self._maybe_fail("upsert")
        self._inner.upsert(chunks)

    def query(self, vector: Vector, k: int, filters: Filter | None = None) -> list[Retrieved]:
        return self._inner.query(vector, k, filters)

    def get(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return self._inner.get(chunk_ids)

    def get_by_source(self, source_id: str) -> list[Chunk]:
        return self._inner.get_by_source(source_id)

    def delete(self, *, source_id: str | None = None, chunk_ids: Sequence[str] | None = None) -> int:
        self._maybe_fail("delete")
        return self._inner.delete(source_id=source_id, chunk_ids=chunk_ids)

    def list_sources(self) -> list[SourceInfo]:
        return self._inner.list_sources()

    def count(self) -> int:
        return self._inner.count()
