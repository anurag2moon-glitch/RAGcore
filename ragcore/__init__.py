"""ragcore — a reusable retrieval layer for RAG.

Public surface:

    from ragcore import Corpus, Retrieved, Chunk, ChunkMeta, SourceInfo, IngestReport, Answer
    from ragcore.embedding import VoyageContextEmbedder
    from ragcore.stores import LanceDBStore
    from ragcore.chunking import StructuralChunker
    from ragcore.generation import ClaudeGenerator

`Corpus` is the only object consuming projects need to hold — it wires
together a Loader, Chunker, Embedder, VectorStore, and (optionally) a
Generator. Each of those is a `typing.Protocol` (see
ragcore.{loaders,chunking,embedding,stores,generation}.base) so any
component can be swapped without touching the others. `Corpus.retrieve()`
returns raw chunks; `Corpus.ask()` additionally synthesizes an answer via a
configured Generator — retrieval works with no Generator at all.

Two optional layers attach in front of retrieval/generation without
changing `Corpus` itself, through the existing `corpus.retriever` /
`corpus.generator` seams:

    from ragcore.graph import build_graph, SqliteGraphStore, ClaudeEntityExtractor
    from ragcore.retrieval.graph import GraphRetriever          # entity-graph-aware retrieval
    from ragcore.concepts import build_concepts, detect_communities, ...
    from ragcore.retrieval.concept import ConceptRetriever      # concept-graph-aware retrieval
    from ragcore.retrieval.mmr import MMRReranker                # diversity reranking
    from ragcore.verification import ClaudeVerifier, VerifyingGenerator  # citation verification
    from ragcore.claims import build_claims, link_claims, SqliteClaimStore  # claim extraction/linking (storage only, no retriever)
    from ragcore.conversation import PersonaChatSession, DebateSession  # session transcript + query rewriting, on top of an existing Corpus

    corpus.retriever = MMRReranker(ConceptRetriever(GraphRetriever(corpus.retriever, ...), ...))
    corpus.generator = VerifyingGenerator(corpus.generator, ClaudeVerifier())
"""

from ragcore.corpus import Corpus
from ragcore.errors import (
    ChunkingError,
    EmbeddingError,
    GenerationError,
    LoaderError,
    RagcoreError,
    StoreError,
)
from ragcore.generation.base import Answer
from ragcore.types import (
    Chunk,
    ChunkGroup,
    ChunkMeta,
    Document,
    Filter,
    IngestProgress,
    IngestReport,
    Retrieved,
    SourceInfo,
    Vector,
)

__all__ = [
    "Corpus",
    "Answer",
    "Chunk",
    "ChunkGroup",
    "ChunkMeta",
    "Document",
    "Filter",
    "IngestProgress",
    "IngestReport",
    "Retrieved",
    "SourceInfo",
    "Vector",
    "RagcoreError",
    "LoaderError",
    "ChunkingError",
    "EmbeddingError",
    "StoreError",
    "GenerationError",
]
