"""GraphRAG layer — scaffold only.

Defines the types and protocols (`GraphStore`, `EntityExtractor`) that a
future graph implementation will satisfy. No graph construction, extraction,
or graph-aware retrieval ships here yet; see ragcore/graph/base.py and
ragcore/graph/types.py for the contracts, and the design doc for how a
`GraphRetriever` will later compose in front of the vector retriever without
any change to ragcore.stores, ragcore.embedding, ragcore.chunking, or
ragcore.loaders.
"""

from ragcore.graph.base import EntityExtractor, GraphStore
from ragcore.graph.types import Entity, GraphContext, Relation

__all__ = [
    "Entity",
    "Relation",
    "GraphContext",
    "GraphStore",
    "EntityExtractor",
]
