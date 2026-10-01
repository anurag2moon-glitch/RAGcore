"""GraphRAG layer: entity/relation extraction, graph construction, and
graph-aware retrieval, attached in front of vector retrieval without any
change to ragcore.stores, ragcore.embedding, ragcore.chunking, or
ragcore.loaders — see the design doc's GraphRAG addendum for the full
rationale.

    from ragcore.graph import (
        Entity, Relation, GraphContext,       # types
        GraphStore, EntityExtractor,          # protocols
        EntityResolver, EmbeddingEntityResolver,
        SqliteGraphStore, ClaudeEntityExtractor,  # default implementations
        build_graph, GraphReport,             # construction pass
    )
    from ragcore.retrieval.graph import GraphRetriever  # graph-aware Retriever
"""

from ragcore.graph.base import EntityExtractor, GraphStore
from ragcore.graph.claude_extractor import ClaudeEntityExtractor
from ragcore.graph.kuzu_store import KuzuGraphStore
from ragcore.graph.pipeline import GraphReport, build_graph
from ragcore.graph.resolver import EmbeddingEntityResolver, EntityResolver
from ragcore.graph.sqlite_store import SqliteGraphStore
from ragcore.graph.types import Entity, GraphContext, Relation

__all__ = [
    "Entity",
    "Relation",
    "GraphContext",
    "GraphStore",
    "EntityExtractor",
    "EntityResolver",
    "EmbeddingEntityResolver",
    "SqliteGraphStore",
    "KuzuGraphStore",
    "ClaudeEntityExtractor",
    "build_graph",
    "GraphReport",
]
