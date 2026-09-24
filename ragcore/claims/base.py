"""Protocols for the claim extraction and linking layer. Nothing in
ragcore.types references ragcore.claims, so — like ragcore.concepts and
ragcore.hierarchy — this package has no import cycle to defer around.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from ragcore.claims.types import ClaimRelation, SourceClaim
from ragcore.concepts.types import Concept
from ragcore.types import Chunk


@runtime_checkable
class ClaimExtractor(Protocol):
    """Extracts atomic factual/philosophical claims from one chapter's worth
    of already-embedded chunks, anchoring each claim to one or more of that
    chapter's already-extracted concepts.

    `concepts` is passed in by `ragcore.claims.pipeline.build_claims`
    (sourced from `ConceptStore.concepts_for_chunks`), not looked up by the
    extractor itself — an extractor mapping claims to concepts *after*
    extraction via chunk-level concept membership would attach every concept
    in a chunk to every claim in it, manufacturing false comparison groups.
    Giving the model the chapter's own concept menu and having it cite which
    concept(s) each claim is about is the only anchoring that's actually
    claim-specific.
    """

    model: str

    def extract(self, *, chunks: Sequence[Chunk], concepts: Sequence[Concept]) -> list[SourceClaim]: ...


@runtime_checkable
class ClaimLinker(Protocol):
    """Given a small candidate batch of claims (already shortlisted by
    ragcore.claims.linking.link_claims — never a whole concept's full claim
    set), returns every agree/conflict/extends relation found among them."""

    model: str

    def link(self, claims: Sequence[SourceClaim]) -> list[ClaimRelation]: ...


@runtime_checkable
class ClaimStore(Protocol):
    """Persists SourceClaims, their concept associations, and
    ClaimRelations; tracks per-chapter extraction checkpoints (for
    build_claims() resumability) and per-concept linking checkpoints (for
    link_claims() resumability).

    `add_chapter_claims` is the one write path for extraction output, and is
    deliberately a single call: a checkpointed pipeline must write claims,
    their concept associations, and the checkpoint row in one transaction,
    or a crash between two separate writes could mark a chapter done with
    associations missing — see ragcore.claims.pipeline.build_claims and
    ragcore.concepts.base.ConceptStore.add_chapter_concepts, which this
    mirrors exactly.
    """

    def add_chapter_claims(
        self,
        *,
        source_id: str,
        context_group_id: str,
        fingerprint: str,
        claims: Sequence[SourceClaim],
    ) -> None: ...

    def checkpoint(self, source_id: str, context_group_id: str) -> str | None:
        """The fingerprint recorded for this chapter's last successful
        (status='done') claim extraction, or None if it has never
        completed (including: never attempted, because the chapter had no
        concepts to anchor claims to yet — see build_claims)."""
        ...

    def mark_chapter_failed(
        self, *, source_id: str, context_group_id: str, fingerprint: str, error: str
    ) -> None: ...

    def claim(self, claim_id: str) -> SourceClaim | None: ...

    def claims_for_concept(self, concept_id: str) -> list[SourceClaim]: ...

    def concept_ids_with_claims(self, *, min_count: int = 2) -> list[str]:
        """Every concept_id reachable from at least `min_count` claims — the
        set ragcore.claims.linking.link_claims iterates over (a concept with
        fewer than 2 claims has nothing to compare)."""
        ...

    def put_relations(self, relations: Sequence[ClaimRelation]) -> None:
        """Merge by content-addressed relation_id (insert-if-absent) —
        never replace. This is what makes ragcore.claims.linking's batched,
        overlapping-candidate-pair calls safe to run without an extra dedup
        pass: two batches producing the same relation_id simply collapse."""
        ...

    def relations_for_claim(self, claim_id: str) -> list[ClaimRelation]:
        """Every relation touching this claim, in either direction."""
        ...

    def link_checkpoint(self, concept_id: str) -> str | None:
        """The fingerprint recorded for this concept's last successful
        linking run, or None if it has never completed."""
        ...

    def mark_concept_linked(self, *, concept_id: str, fingerprint: str) -> None: ...

    def remove_chunk_ids(self, chunk_ids: Sequence[str]) -> None:
        """Deletes every claim originating from the given chunk_ids, along
        with its concept associations and every relation touching it in
        EITHER direction — see ragcore.versioning.sync. A relation is
        deleted the moment either endpoint claim is deleted, even though the
        other endpoint (and its own source) may be untouched — the same
        rule ragcore.concepts.sqlite_store.SqliteConceptStore.remove_chunk_ids
        applies when an edge's endpoint concept vanishes."""
        ...

    def remove_source(self, source_id: str) -> None:
        """All claims belonging to `source_id`, pruned via
        remove_chunk_ids, plus that source's own extraction checkpoints.
        Unlike ConceptStore.remove_source, this does not need a chunk_id
        prefix-matching trick — SourceClaim stores source_id as a real
        column."""
        ...

    def backup_to(self, path: str | Path) -> None: ...

    def restore_from_backup(self, path: str | Path) -> None: ...
