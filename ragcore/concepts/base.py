"""Protocols for the concept/theme layer.

Unlike ragcore.graph, nothing in ragcore.types references ragcore.concepts
(there is no Concept-typed field on any core dataclass — a community summary
reaches retrieval as a synthetic Chunk instead, see
ragcore.concepts.communities and ragcore.retrieval.concept), so this package
has no import cycle to defer around: Chunk/Corpus can be imported directly.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ragcore.concepts.types import Concept, ConceptEdge, CommunitySummary
from ragcore.types import Chunk


@runtime_checkable
class ConceptStore(Protocol):
    """Persists concepts, concept-chunk links, concept co-occurrence edges,
    and community summaries; tracks per-chapter extraction checkpoints for
    build_concepts() resumability.

    `add_chapter_concepts` is the one write path for extraction output, and
    it is deliberately a single call (not add_concepts()/add_edges()
    separately, unlike ragcore.graph.base.GraphStore's add_entities/
    add_relations split): a checkpointed pipeline must write concepts,
    edges, and the checkpoint row in one transaction, or a crash between two
    separate writes could mark a chapter done with edges missing — see
    ragcore.concepts.pipeline.build_concepts and CLAUDE.md.
    """

    def add_chapter_concepts(
        self,
        *,
        source_id: str,
        context_group_id: str,
        fingerprint: str,
        concepts: Sequence[Concept],
        edges: Sequence[ConceptEdge],
    ) -> None: ...

    def checkpoint(self, source_id: str, context_group_id: str) -> str | None:
        """The fingerprint recorded for this chapter's last successful
        (status='done') extraction, or None if it has never completed."""
        ...

    def mark_chapter_failed(
        self, *, source_id: str, context_group_id: str, fingerprint: str, error: str
    ) -> None:
        """Records a failed extraction attempt for introspection. Resume
        logic in build_concepts() never actually consults this — the
        absence of a 'done' checkpoint is already sufficient to retry a
        chapter — but a caller inspecting the store directly can see why a
        chapter is still pending."""
        ...

    def concept(self, concept_id: str) -> Concept | None: ...

    def all_concepts(self) -> list[Concept]: ...

    def concepts_for_chunks(self, chunk_ids: Sequence[str]) -> dict[str, list[Concept]]: ...

    def chunks_for_concepts(self, concept_ids: Sequence[str]) -> dict[str, list[str]]: ...

    def all_edges(self) -> list[ConceptEdge]:
        """Every concept co-occurrence edge — the input to
        ragcore.concepts.communities.detect_communities."""
        ...

    def concept_vector(self, concept_id: str) -> tuple[float, ...] | None: ...

    def community_run_fingerprint(self) -> str | None:
        """Fingerprint of the last successful detect_communities() run, or
        None if it has never run. See ragcore.concepts.communities."""
        ...

    def set_communities(
        self, *, run_fingerprint: str, assignments: dict[str, str]
    ) -> None:
        """Record which community_id each concept_id belongs to for this
        clustering run. `assignments` is concept_id -> community_id."""
        ...

    def community_id_for_concept(self, concept_id: str) -> str | None: ...

    def concept_ids_for_community(self, community_id: str) -> list[str]: ...

    def all_community_ids(self) -> list[str]: ...

    def community_summary(self, community_id: str) -> CommunitySummary | None: ...

    def put_community_summary(self, summary: CommunitySummary) -> None: ...

    def communities_for_concepts(
        self, concept_ids: Sequence[str]
    ) -> list[CommunitySummary]:
        """Every distinct community summary reachable from any of the given
        concept ids — a concept without a current community (or without a
        summary yet) simply contributes nothing."""
        ...

    def remove_chunk_ids(self, chunk_ids: Sequence[str]) -> None:
        """Prunes the given chunk_ids from every concept's/edge's
        provenance, deleting any concept or edge left with zero remaining
        chunk associations. See ragcore.versioning.sync."""
        ...

    def remove_source(self, source_id: str) -> None:
        """All chunk associations belonging to `source_id`, pruned via
        remove_chunk_ids, plus that source's own extraction checkpoints.
        See ragcore.versioning.sync."""
        ...

    def prune_orphan_summaries(self) -> None:
        """Deletes community summaries for communities no longer present —
        called at the end of detect_communities()."""
        ...


@runtime_checkable
class ConceptExtractor(Protocol):
    """Extracts abstract concepts/themes and their co-occurrence edges from
    one chapter's worth of already-embedded chunks."""

    model: str

    def extract(self, chunks: Sequence[Chunk]) -> tuple[list[Concept], list[ConceptEdge]]: ...


@runtime_checkable
class ConceptResolver(Protocol):
    def resolve(self, candidate: Concept, existing: Sequence[Concept]) -> str | None:
        """Return the concept_id of an existing concept `candidate` should
        be merged into, or None if it's a genuinely new, distinct concept."""
        ...


@runtime_checkable
class ConceptMapper(Protocol):
    """Maps a raw query to the ids of relevant concepts already in the
    graph. See ragcore.concepts.mapper: EmbeddingConceptMapper (default, no
    LLM call) and ClaudeConceptMapper (optional refinement on top of it)."""

    def map(self, query: str, concepts: Sequence[Concept]) -> list[str]: ...


@runtime_checkable
class CommunitySummarizer(Protocol):
    model: str

    def summarize(
        self,
        *,
        community_id: str,
        concepts: Sequence[Concept],
        chunks: Sequence[Chunk],
    ) -> CommunitySummary: ...
