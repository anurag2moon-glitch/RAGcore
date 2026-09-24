"""Protocols for the chunk -> section -> chapter -> book summarization
hierarchy. Nothing in ragcore.types references ragcore.hierarchy, so — like
ragcore.concepts — this package has no import cycle to defer around.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ragcore.hierarchy.types import HierarchyLevel, HierarchyNode


@runtime_checkable
class HierarchySummarizer(Protocol):
    model: str

    def summarize(
        self,
        *,
        node_id: str,
        level: HierarchyLevel,
        source_id: str,
        label: str,
        child_texts: Sequence[str],
        child_ids: Sequence[str],
        source_chunk_ids: Sequence[str],
    ) -> HierarchyNode:
        """Synthesizes one node's summary from its children's texts (chunk
        texts at the section level, child summaries at chapter/book level).
        Returns a HierarchyNode with `vector=None` — build_hierarchy embeds
        the summary and fills that in afterward, so a HierarchySummarizer
        implementation never needs an Embedder of its own (the same
        discipline ragcore.concepts.communities.build_community_summaries
        already follows for CommunitySummarizer)."""
        ...


@runtime_checkable
class HierarchyStore(Protocol):
    """Persists HierarchyNodes and tracks per-node checkpoints for
    build_hierarchy()'s resumability — same single-transaction-with-
    checkpoint shape as ragcore.concepts.base.ConceptStore.add_chapter_concepts.
    """

    def add_node(self, node: HierarchyNode, *, fingerprint: str) -> None: ...

    def checkpoint(self, node_id: str) -> str | None:
        """The fingerprint recorded for this node's last successful build,
        or None if it has never completed."""
        ...

    def mark_node_failed(self, *, node_id: str, fingerprint: str, error: str) -> None: ...

    def node(self, node_id: str) -> HierarchyNode | None: ...

    def nodes_for_source(self, source_id: str) -> list[HierarchyNode]: ...

    def nodes_by_level(self, level: HierarchyLevel) -> list[HierarchyNode]: ...

    def all_nodes(self) -> list[HierarchyNode]: ...

    def remove_source(self, source_id: str) -> None: ...
