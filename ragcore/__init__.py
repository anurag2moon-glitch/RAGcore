"""ragcore — a reusable retrieval layer for RAG.

Public surface:

    from ragcore import Corpus, Retrieved, Chunk, ChunkMeta, SourceInfo, IngestReport
    from ragcore.embedding import VoyageContextEmbedder
    from ragcore.stores import LanceDBStore
    from ragcore.chunking import StructuralChunker

`Corpus` is the only object consuming projects need to hold — it wires
together a Loader, Chunker, Embedder, and VectorStore. Each of those is a
`typing.Protocol` (see ragcore.{loaders,chunking,embedding,stores}.base) so
any component can be swapped without touching the others. A GraphRAG layer
is scaffolded in ragcore.graph (types + protocols only, no implementation
yet) and will attach in front of retrieval without changing this surface —
see the design notes in the project plan for how.
"""

from ragcore.corpus import Corpus
from ragcore.errors import (
    ChunkingError,
    EmbeddingError,
    LoaderError,
    RagcoreError,
    StoreError,
)
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
]
