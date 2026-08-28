"""Data types for the GraphRAG layer.

This module is scaffold: it defines the shapes the graph layer will use so
that `ragcore.types.Retrieved.graph` type-checks and so a future
implementation has a fixed contract to build against. No graph construction
or extraction logic lives here or anywhere in `ragcore.graph` yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Entity:
    """A named entity extracted from one or more chunks."""

    entity_id: str
    name: str
    entity_type: str
    """Open vocabulary (e.g. "person", "concept", "organization") — not an
    enum, since extractors and domains will want their own taxonomies."""

    source_chunk_ids: tuple[str, ...]
    """Chunks this entity was extracted from. The join key back into the
    vector store — both sides key on Chunk.chunk_id."""

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Relation:
    """A directed relation between two entities, attributed to a chunk."""

    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    source_chunk_ids: tuple[str, ...]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphContext:
    """Graph provenance attached to a retrieved chunk: which entities tied
    it to the query, and the traversal path that reached it."""

    entities: tuple[str, ...]
    path: tuple[str, ...]
