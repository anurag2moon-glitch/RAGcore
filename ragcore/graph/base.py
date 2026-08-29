"""Protocols for the GraphRAG layer.

`GraphStore` and `EntityExtractor` are the entire integration seam between
the graph layer and the rest of ragcore: both operate purely in terms of
`Chunk.chunk_id` strings and never import anything from `ragcore.stores` or
`ragcore.embedding`. `GraphRetriever` (`ragcore/retrieval/graph.py`)
composes a `Retriever` with a `GraphStore` by reading chunks back out of a
`Corpus` via `VectorStore.get()`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

from ragcore.graph.types import Entity, Relation

if TYPE_CHECKING:
    # Deferred to break the import cycle: ragcore.types imports
    # ragcore.graph.types (for GraphContext), and importing any submodule
    # of ragcore.graph runs ragcore/graph/__init__.py first, which reaches
    # this module. `from __future__ import annotations` above means Chunk
    # is only ever needed as a string annotation, never at runtime.
    from ragcore.types import Chunk


@runtime_checkable
class GraphStore(Protocol):
    """Persists entities and relations, and answers traversal queries.

    `entities_for_chunks` / `chunks_for_entities` are the join: they let a
    retriever step from vector-search hits into the graph and back into
    chunk ids the vector store can hydrate with `VectorStore.get()`.
    """

    def add_entities(self, entities: Sequence[Entity]) -> None: ...

    def add_relations(self, relations: Sequence[Relation]) -> None: ...

    def neighbors(
        self,
        entity_ids: Sequence[str],
        *,
        hops: int = 1,
        rel_types: Sequence[str] | None = None,
    ) -> list[Relation]: ...

    def entities_for_chunks(self, chunk_ids: Sequence[str]) -> dict[str, list[Entity]]: ...

    def chunks_for_entities(self, entity_ids: Sequence[str]) -> dict[str, list[str]]: ...

    def entities_by_type(self, entity_type: str) -> list[Entity]:
        """All known entities of one type. Used by entity resolution to find
        merge candidates — see ragcore.graph.resolver."""
        ...


@runtime_checkable
class EntityExtractor(Protocol):
    """Extracts entities and relations from already-embedded chunks."""

    def extract(self, chunks: Sequence[Chunk]) -> tuple[list[Entity], list[Relation]]: ...
