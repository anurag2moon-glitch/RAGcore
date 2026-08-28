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
configured Generator — retrieval works with no Generator at all. A GraphRAG
layer is scaffolded in ragcore.graph (types + protocols only, no
implementation yet) and will attach in front of retrieval without changing
this surface — see the design notes in the project plan for how.
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
