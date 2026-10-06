"""Core data types shared across the ingestion, embedding, storage, and
retrieval layers.

Everything here is a frozen dataclass. Chunks and their metadata flow
loader -> chunker -> embedder -> store -> retriever unchanged; nothing
downstream mutates what an upstream stage produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ragcore.graph.types import GraphContext

SourceType = Literal["pdf", "epub", "docx", "txt", "md", "html"]
"""What a Loader can produce. Closed — every Loader implementation emits one
of these six, and Document.source_type / SourceInfo.source_type stay
scoped to exactly this vocabulary."""

ChunkOrigin = SourceType | Literal["synthetic"]
"""What a Chunk's ChunkMeta.source_type may carry — a superset of
SourceType. "synthetic" marks a chunk ragcore generated itself rather than
extracted from a file (e.g. a community summary surfaced as retrievable
context by ragcore.retrieval.concept.ConceptRetriever — see
ragcore.concepts.communities). No Loader ever produces "synthetic"; keeping
it off SourceType itself means that invariant is enforced by the type
checker, not just a docstring, and a VectorStore can reject it on upsert
(see LanceDBStore.upsert) since a synthetic chunk is never meant to be
persisted as if it were real source material."""

IngestStage = Literal["loading", "chunking", "embedding", "storing", "done"]

# A single embedding vector. Kept as a plain list[float] rather than a numpy
# array at this boundary so the protocols don't force a numpy dependency on
# every implementation (e.g. a pure-Python fake store in tests).
Vector = list[float]

# Store-agnostic filter DSL. Keys are ChunkMeta field names; values are
# either a literal (equality), a list (membership), or a dict with
# "gte"/"gt"/"lte"/"lt" range operators. Each VectorStore adapter translates
# this into its own query dialect.
Filter = dict[str, Any]


# --------------------------------------------------------------------------
# Loader output
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Block:
    """One unit of extracted text with its structural position.

    Blocks are flat and ordered, not a tree: each block carries a full
    breadcrumb (`section_path`) to its position in the document's structure
    rather than being nested inside a section object. This keeps the Loader
    contract small and makes chunking a simple linear fold over `blocks`.
    """

    text: str
    page: int | None
    section_path: tuple[str, ...]
    page_hash: str | None = None
    """sha256(text) of this block's own content, set only by loaders with a
    real page concept (PdfLoader — see ragcore/loaders/pdf.py) and left
    None everywhere else, matching `page` itself. Lets
    ragcore.chunking.structural.StructuralChunker tag a chunk that doesn't
    cross a page boundary with ChunkMeta.extra["page_hash"], giving
    content-hash dedupe at the page granularity (alongside the existing
    file-level `source_id` and the chunker's own per-chunk identity) —
    e.g. detecting that a re-scanned copy of the same book has an
    unchanged page without re-embedding it."""


@dataclass(frozen=True)
class Document:
    """A single ingested file, fully extracted and structurally tagged."""

    source_id: str
    """Content-addressed id (sha256 of file bytes). Stable across re-ingestion
    of the same file, which is what lets a later-built graph's edges stay
    valid against re-ingested chunks."""

    path: str
    title: str | None
    source_type: SourceType
    blocks: list[Block]
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Chunking output
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkMeta:
    """Metadata carried by every chunk from ingestion through retrieval."""

    # --- provenance ---
    source_id: str
    source_path: str
    source_title: str | None
    source_type: ChunkOrigin

    # --- book structure ---
    section_path: tuple[str, ...]
    chapter_title: str | None
    """Raw chapter title text, taken verbatim from the source's own
    structure metadata (EPUB TOC label, PDF outline entry, DOCX/MD heading
    text). Display-only — depending on source quality this can be noisy or
    malformed (e.g. some Project Gutenberg EPUB editions concatenate an
    illustration caption onto the chapter heading in the TOC label itself).
    Never key retrieval or filtering off this field; use `chapter_index`."""
    chapter_index: int | None
    """Ordinal position of this chunk's chapter among all structural groups
    encountered in document order (spine order for EPUB, page/block order
    otherwise) — stable and clean regardless of how noisy `chapter_title`
    is. Not necessarily the book's own printed chapter number: front matter
    or title-page groups before the first real chapter consume index slots
    too, so "Chapter IV" may have chapter_index=5, not 4. Safe to filter
    and range-query on (e.g. `{"chapter_index": {"gte": 7}}`)."""
    section: str | None
    page_start: int | None
    page_end: int | None

    # --- position & identity ---
    chunk_index: int
    context_group_id: str
    char_start: int
    char_end: int
    token_count: int

    # --- multi-tenancy (Phase 7 foundation) ---
    tenant_id: str | None = None
    """Opt-in, additive, default `None` (docs/ARCHITECTURE.md Phase 7 gap
    list: "start with tenant_id propagation through ChunkMeta and every
    store schema, since every other Phase-7 item (quotas, isolation,
    metering) depends on that field existing first"). `None` means
    single-tenant mode — every existing caller that never passes
    `tenant_id=...` anywhere sees identical behavior to before this field
    existed; no filter is applied, no isolation is enforced, nothing
    changes. A caller opting into multi-tenant isolation sets it via
    `Corpus.ingest(..., tenant_id=...)` and is responsible for passing a
    matching `filters={"tenant_id": ...}` on every `retrieve()`/`ask()`
    call — this field makes tenant-scoped filtering *possible* at the
    storage layer; it does not itself enforce isolation (no hosting-layer
    guard against a caller forgetting the filter — see
    docs/adr/0006-tenant-id-foundation-not-enforcement.md)."""

    # --- extensibility ---
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A chunk of text plus its metadata, with an optional embedding vector.

    `vector` is None as produced by a Chunker, populated by an Embedder, and
    always present by the time a VectorStore persists it.
    """

    chunk_id: str
    """f"{source_id}:{chunk_index}" — stable and content-addressed, so
    re-ingesting the same file reproduces identical ids."""

    text: str
    meta: ChunkMeta
    vector: Vector | None = None


@dataclass(frozen=True)
class ChunkGroup:
    """A set of chunks embedded together, sharing one contextualization
    window (e.g. one chapter).

    Chunkers return groups rather than a flat chunk list specifically so
    contextualized embedding is expressible: a group is exactly the unit an
    Embedder.embed_groups() call contextualizes jointly. A chunker for a
    non-contextual model can simply return one group per chunk.
    """

    group_id: str
    chunks: list[Chunk]


# --------------------------------------------------------------------------
# Retrieval output
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Retrieved:
    """One retrieval result: a chunk, its score, and optional graph
    provenance.

    `graph` is typed against ragcore.graph.types.GraphContext, which exists
    today purely as a scaffold — no graph implementation ships yet. Until
    one does, this field stays None. Consuming code that never reads it is
    unaffected when a GraphRetriever is later wired in front of the vector
    retriever (see ragcore.graph).
    """

    chunk: Chunk
    score: float
    graph: GraphContext | None = None


# --------------------------------------------------------------------------
# Corpus-level reporting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceInfo:
    """Summary of one ingested source, as returned by Corpus.sources()."""

    source_id: str
    path: str
    title: str | None
    source_type: SourceType
    chunk_count: int


@dataclass(frozen=True)
class IngestProgress:
    """Progress callback payload for Corpus.ingest()/ingest_many()."""

    path: str
    stage: IngestStage
    chunks_done: int
    chunks_total: int


@dataclass(frozen=True)
class IngestReport:
    """Summary returned by Corpus.ingest()/ingest_many()."""

    source_ids: list[str]
    chunks_ingested: int
    sources_replaced: list[str]
    errors: list[tuple[str, str]]
    """(path, error message) pairs for files that failed to ingest, when
    ingest_many() is used in best-effort mode."""
