"""Data types for the chunk -> section -> chapter -> book summarization
hierarchy.

Mirrors ragcore.concepts.types.CommunitySummary's shape, but a HierarchyNode's
id is positional rather than content-addressed: unlike a concept or
community (which can legitimately recur across completely different
sources and benefits from content-addressed merging), a hierarchy node's
identity is inherently tied to one exact position in one book's own table
of contents — see ragcore.hierarchy.pipeline's module docstring for the
exact id scheme and the reasoning. Resumability instead comes entirely from
HierarchyStore's per-node checkpoint fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

HierarchyLevel = Literal["section", "chapter", "book"]


@dataclass(frozen=True)
class HierarchyNode:
    node_id: str
    """Positional, e.g. f"hierarchy:chapter:{source_id}:{chapter_index}" —
    see ragcore.hierarchy.pipeline for the exact scheme per level. Stable
    regardless of content changes; ragcore.hierarchy.base.HierarchyStore's
    per-node checkpoint fingerprint (not the id) is what decides whether a
    node needs rebuilding."""

    level: HierarchyLevel
    source_id: str
    label: str
    """chapter_title / section name / book title — display text, taken from
    the same source structure metadata ChunkMeta already carries."""

    summary: str
    child_ids: tuple[str, ...]
    """chunk_ids (section level) or child HierarchyNode.node_ids
    (chapter/book level)."""

    source_chunk_ids: tuple[str, ...]
    """Every leaf chunk_id under this node, regardless of level — the
    provenance a citation or verification pass would need, without having
    to walk child_ids recursively."""

    model: str
    prompt_version: str
    created_at: str
    vector: tuple[float, ...] | None = None
    """The summary text's own embedding, computed once when the node is
    built and persisted alongside it — never invented at retrieval time,
    the same discipline ragcore.concepts.types.CommunitySummary.vector
    follows, for the same reason (ragcore.retrieval.hierarchy.HierarchyRetriever
    needs a real vector to score a node against a query on the same scale
    every other retrieval layer uses)."""
