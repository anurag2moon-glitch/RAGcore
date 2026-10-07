"""TenantScopedCorpus: a Corpus wrapper that always applies one fixed
tenant_id, rather than leaving every call site responsible for remembering
to pass it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from ragcore.errors import TenancyError
from ragcore.tenancy.types import TenantDeletionReport
from ragcore.types import Chunk, Filter, IngestProgress, IngestReport, Retrieved, SourceInfo

if TYPE_CHECKING:
    from ragcore.corpus import Corpus
    from ragcore.generation.base import Answer, Generator

logger = logging.getLogger("ragcore.tenancy")


class TenantScopedCorpus:
    """Wraps an existing `Corpus` plus one fixed `tenant_id`. Every
    ingest/retrieve/ask/get/sources/remove/count call below forces that
    tenant_id rather than trusting the caller to supply it correctly (or
    at all) on every call — the gap `docs/adr/0006-tenant-id-foundation-
    not-enforcement.md` names as open. A caller-supplied `tenant_id=...`
    argument or `filters={"tenant_id": ...}` that disagrees with this
    instance's own `tenant_id` raises `TenancyError` rather than being
    silently overridden or silently honored — either of those would hide
    a real bug (a caller that thinks it's operating as a different
    tenant than the one this wrapper was actually constructed for).

    **Known scope limit**: this only enforces isolation for the
    vector-store-level retrieval path (`VectorRetriever`/`HybridRetriever`,
    both of which call `VectorStore.query(vector, k, filters)`, the one
    place `tenant_id` is filterable today — see `LanceDBStore`). A
    `corpus.retriever` with `GraphRetriever`/`ConceptRetriever`/
    `HierarchyRetriever`/`IterativeRetriever` layered on top (see
    `ragcore.retrieval.stack.default_stack`) reads its *own* store
    (`SqliteGraphStore`/`KuzuGraphStore`, `SqliteConceptStore`, the
    hierarchy store), none of which carry `tenant_id` yet
    (`docs/ARCHITECTURE.md` Phase 7 gap item 28, still open) — those
    layers are not tenant-filtered by this wrapper and can surface
    cross-tenant graph-expanded or synthetic-summary chunks. Do not rely
    on this wrapper alone for isolation if any of those optional layers
    are wired onto the underlying `Corpus`.

    `ingest()`/`ingest_many()`'s `replace=True` default deletes a
    source's existing chunks by `source_id` (a content hash) before
    re-inserting — this wrapper does not additionally check that an
    existing source being replaced already belonged to this tenant,
    since `Corpus.ingest()` itself has no notion of "whose" a
    content-addressed source_id is; two tenants ingesting byte-identical
    files would collide on `replace=True` with or without this wrapper
    (a pre-existing `Corpus` characteristic, not something introduced
    here — see the `source_id` content-addressing note in `CLAUDE.md`).
    """

    def __init__(self, corpus: Corpus, tenant_id: str) -> None:
        if not tenant_id:
            raise TenancyError("tenant_id must be a non-empty string")
        self.corpus = corpus
        self.tenant_id = tenant_id

    # -- ingestion -----------------------------------------------------

    def ingest(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        replace: bool = True,
        on_progress: Callable[[IngestProgress], None] | None = None,
    ) -> IngestReport:
        return self.corpus.ingest(
            path,
            metadata=metadata,
            tenant_id=self._resolve_tenant_id(tenant_id),
            replace=replace,
            on_progress=on_progress,
        )

    def ingest_many(
        self,
        paths: Sequence[str | Path],
        *,
        metadata: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        replace: bool = True,
        on_progress: Callable[[IngestProgress], None] | None = None,
        max_workers: int = 4,
        max_retries: int = 0,
    ) -> IngestReport:
        return self.corpus.ingest_many(
            paths,
            metadata=metadata,
            tenant_id=self._resolve_tenant_id(tenant_id),
            replace=replace,
            on_progress=on_progress,
            max_workers=max_workers,
            max_retries=max_retries,
        )

    # -- retrieval -------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        k: int = 10,
        filters: Filter | None = None,
        min_score: float | None = None,
    ) -> list[Retrieved]:
        return self.corpus.retrieve(
            query, k=k, filters=self._merge_filters(filters), min_score=min_score
        )

    def ask(
        self,
        query: str,
        *,
        k: int = 10,
        filters: Filter | None = None,
        generator: Generator | None = None,
    ) -> Answer:
        return self.corpus.ask(
            query, k=k, filters=self._merge_filters(filters), generator=generator
        )

    # -- corpus management -----------------------------------------------

    def get(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        chunks = self.corpus.get(chunk_ids)
        foreign = [c.chunk_id for c in chunks if c.meta.tenant_id != self.tenant_id]
        if foreign:
            raise TenancyError(
                f"chunk_ids {foreign} do not belong to tenant {self.tenant_id!r}"
            )
        return chunks

    def sources(self) -> list[SourceInfo]:
        """Filters `corpus.sources()` down to sources whose chunks carry
        this tenant_id. One `get_by_source()` call per source (checking
        only its first chunk, since every chunk of one `ingest()` call
        shares one tenant_id by construction) — O(this tenant's own
        chunk count) across all tenants' sources combined, not a cheap
        indexed lookup, since no VectorStore today exposes a
        tenant-filtered `list_sources()`. Fine for a moderate source
        count; a store with many tenants and many sources each would
        want that added to the VectorStore protocol instead."""
        scoped = []
        for info in self.corpus.sources():
            chunks = self.corpus.store.get_by_source(info.source_id)
            if chunks and chunks[0].meta.tenant_id == self.tenant_id:
                scoped.append(info)
        return scoped

    def remove(self, source_id: str) -> int:
        chunks = self.corpus.store.get_by_source(source_id)
        if not chunks:
            return 0
        if chunks[0].meta.tenant_id != self.tenant_id:
            raise TenancyError(
                f"source_id {source_id!r} does not belong to tenant {self.tenant_id!r}"
            )
        return self.corpus.remove(source_id)

    def count(self) -> int:
        return sum(info.chunk_count for info in self.sources())

    def hard_delete(
        self,
        *,
        graph_store: Any = None,
        concept_store: Any = None,
        claim_store: Any = None,
        context_store: Any = None,
        hierarchy_store: Any = None,
        lexical_store: Any = None,
    ) -> TenantDeletionReport:
        """Deletes every one of this tenant's chunks from the vector
        store, plus every row derived from them in whichever optional
        auxiliary stores a caller actually passes — `Corpus` itself
        holds no reference to a graph/concept/claim/context/hierarchy/
        lexical store (those are built and held externally, see
        `CLAUDE.md`), so this method can't discover them on its own.

        Reuses each store's own existing `remove_source()` (or, for
        `lexical_store`, `delete(source_id=...)`) completely unchanged —
        no new removal logic, no schema change, anywhere. That's what
        makes this safe for `graph_store`/`concept_store` too, despite
        their content-addressed entity/relation/concept ids not being
        tenant-scoped (`docs/adr/0007-tenant-id-on-remaining-stores-two-tiers.md`):
        removing a chunk_id's association from an entity/concept only
        ever subtracts *this* tenant's own provenance from a row — it
        cannot touch another tenant's association with that same row,
        even one sharing the same id, because `remove_source`/
        `remove_chunk_ids` key strictly off chunk_id, never off the
        entity/concept id's own merged content. An entity/concept is
        only ever deleted outright once every chunk_id referencing it
        (from any tenant) is gone, which is the same "no other tenant
        was ever involved to begin with" case a single-tenant deployment
        already relies on.

        Source enumeration is this tenant's own (`self.sources()`), so a
        store holding other tenants' data untouched by this call is
        guaranteed by construction, not by trusting each store's own
        `remove_source` to filter correctly."""
        source_ids = tuple(info.source_id for info in self.sources())
        chunks_deleted = sum(self.corpus.remove(source_id) for source_id in source_ids)

        for store in (graph_store, concept_store, claim_store, context_store, hierarchy_store):
            if store is None:
                continue
            for source_id in source_ids:
                store.remove_source(source_id)

        if lexical_store is not None:
            for source_id in source_ids:
                lexical_store.delete(source_id=source_id)

        logger.info(
            "hard_delete: tenant=%r removed %d sources, %d chunks",
            self.tenant_id, len(source_ids), chunks_deleted,
        )
        return TenantDeletionReport(
            tenant_id=self.tenant_id, source_ids=source_ids, chunks_deleted=chunks_deleted
        )

    # -- internal ----------------------------------------------------------

    def _resolve_tenant_id(self, tenant_id: str | None) -> str:
        if tenant_id is not None and tenant_id != self.tenant_id:
            raise TenancyError(
                f"tenant_id={tenant_id!r} conflicts with this TenantScopedCorpus's "
                f"tenant_id {self.tenant_id!r}"
            )
        return self.tenant_id

    def _merge_filters(self, filters: Filter | None) -> Filter:
        if filters and "tenant_id" in filters and filters["tenant_id"] != self.tenant_id:
            raise TenancyError(
                f"filters={{'tenant_id': {filters['tenant_id']!r}}} conflicts with "
                f"this TenantScopedCorpus's tenant_id {self.tenant_id!r}"
            )
        return {**(filters or {}), "tenant_id": self.tenant_id}
