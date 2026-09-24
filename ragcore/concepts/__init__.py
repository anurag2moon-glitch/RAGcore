"""Concept/theme layer: abstract-idea extraction, a concept co-occurrence
graph, community detection, and precomputed community summaries — attached
in front of retrieval the same way ragcore.graph is, without any change to
ragcore.stores, ragcore.embedding, ragcore.chunking, or ragcore.loaders.

    from ragcore.concepts import (
        Concept, ConceptEdge, CommunitySummary,             # types
        ConceptStore, ConceptExtractor, ConceptResolver,    # protocols
        ConceptMapper, CommunitySummarizer,
        EmbeddingConceptResolver, SqliteConceptStore,       # default implementations
        ClaudeConceptExtractor, EmbeddingConceptMapper,
        ClaudeConceptMapper, ClaudeCommunitySummarizer,
        build_concepts, ConceptReport, ConceptProgress,     # construction pass
        detect_communities, CommunityReport,                # community detection
        build_community_summaries,
    )
    from ragcore.retrieval.concept import ConceptRetriever  # concept-aware Retriever
    from ragcore.retrieval.mmr import MMRReranker           # diversity reranking
"""

from ragcore.concepts.base import (
    CommunitySummarizer,
    ConceptExtractor,
    ConceptMapper,
    ConceptStore,
)
from ragcore.concepts.claude_extractor import ClaudeConceptExtractor
from ragcore.concepts.communities import (
    ClaudeCommunitySummarizer,
    CommunityReport,
    build_community_summaries,
    detect_communities,
)
from ragcore.concepts.mapper import ClaudeConceptMapper, EmbeddingConceptMapper
from ragcore.concepts.pipeline import ConceptProgress, ConceptReport, build_concepts
from ragcore.concepts.resolver import ConceptResolver, EmbeddingConceptResolver
from ragcore.concepts.sqlite_store import SqliteConceptStore
from ragcore.concepts.types import CommunitySummary, Concept, ConceptEdge

__all__ = [
    "Concept",
    "ConceptEdge",
    "CommunitySummary",
    "ConceptStore",
    "ConceptExtractor",
    "ConceptResolver",
    "ConceptMapper",
    "CommunitySummarizer",
    "EmbeddingConceptResolver",
    "SqliteConceptStore",
    "ClaudeConceptExtractor",
    "EmbeddingConceptMapper",
    "ClaudeConceptMapper",
    "ClaudeCommunitySummarizer",
    "build_concepts",
    "ConceptReport",
    "ConceptProgress",
    "detect_communities",
    "CommunityReport",
    "build_community_summaries",
]
