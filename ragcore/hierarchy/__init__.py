"""Chunk -> section -> chapter -> book summarization hierarchy, attached in
front of retrieval the same way ragcore.concepts is — a separate
checkpointed pass, run after ingest(), never inside it.

    from ragcore.hierarchy import (
        HierarchyLevel, HierarchyNode,          # types
        HierarchySummarizer, HierarchyStore,    # protocols
        ClaudeHierarchySummarizer, SqliteHierarchyStore,  # default implementations
        build_hierarchy, HierarchyReport, HierarchyProgress,
    )
    from ragcore.retrieval.hierarchy import HierarchyRetriever  # structural-zoom-aware Retriever
"""

from ragcore.hierarchy.base import HierarchyStore, HierarchySummarizer
from ragcore.hierarchy.claude import ClaudeHierarchySummarizer
from ragcore.hierarchy.pipeline import HierarchyProgress, HierarchyReport, build_hierarchy
from ragcore.hierarchy.sqlite_store import SqliteHierarchyStore
from ragcore.hierarchy.types import HierarchyLevel, HierarchyNode

__all__ = [
    "HierarchyLevel",
    "HierarchyNode",
    "HierarchySummarizer",
    "HierarchyStore",
    "ClaudeHierarchySummarizer",
    "SqliteHierarchyStore",
    "build_hierarchy",
    "HierarchyReport",
    "HierarchyProgress",
]
