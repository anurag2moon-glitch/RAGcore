"""Corpus — the single object consuming projects hold.

Orchestration only: ingestion wires Loader -> Chunker -> Embedder -> Store,
retrieval delegates to a Retriever. No I/O logic of its own beyond calling
into those components, so swapping any one of them never touches this file.
"""

from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ragcore.chunking.base import Chunker
from ragcore.chunking.structural import StructuralChunker
from ragcore.embedding.base import Embedder
from ragcore.loaders import LoaderRegistry
from ragcore.retrieval.base import Retriever
from ragcore.retrieval.vector import VectorRetriever
from ragcore.stores.base import VectorStore
from ragcore.types import (
    Chunk,
    ChunkGroup,
    Filter,
    IngestProgress,
    IngestReport,
    Retrieved,
    SourceInfo,
)


class Corpus:
    """A book collection: ingest files into it, retrieve chunks from it.

    `retriever` is a public, reassignable attribute — this is the seam the
    future graph layer uses. `Corpus.retriever = GraphRetriever(corpus.retriever,
    graph_store, corpus)` swaps in graph-aware retrieval without changing any
    `corpus.retrieve(...)` call site.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        chunker: Chunker | None = None,
        loaders: LoaderRegistry | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.chunker = chunker or StructuralChunker()
        self.loaders = loaders or LoaderRegistry()
        self.retriever: Retriever = VectorRetriever(embedder, store)

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ) -> "Corpus":
        """One-liner entry point: opens (or creates) an on-disk corpus at
        `path`, defaulting to voyage-context-4 over an embedded LanceDB
        table sized to the embedder's output dimension."""
        from ragcore.embedding.voyage import VoyageContextEmbedder
        from ragcore.stores.lancedb import LanceDBStore

        embedder = embedder or VoyageContextEmbedder()
        store = store or LanceDBStore(path, dim=embedder.dim)
        return cls(store, embedder)

    # -- ingestion -----------------------------------------------------

    def ingest(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        replace: bool = True,
        on_progress: Callable[[IngestProgress], None] | None = None,
    ) -> IngestReport:
        path = Path(path)
        loader = self.loaders.get_loader(path)

        self._report(on_progress, path, "loading", 0, 0)
        doc = loader.load(path)

        self._report(on_progress, path, "chunking", 0, 0)
        groups = self.chunker.chunk(doc)
        total_chunks = sum(len(g.chunks) for g in groups)

        self._report(on_progress, path, "embedding", 0, total_chunks)
        embedded_groups = self._embed_groups(groups)

        chunks: list[Chunk] = []
        for group in embedded_groups:
            for chunk in group.chunks:
                if metadata:
                    chunk = dataclasses.replace(
                        chunk,
                        meta=dataclasses.replace(
                            chunk.meta, extra={**chunk.meta.extra, **metadata}
                        ),
                    )
                chunks.append(chunk)

        replaced: list[str] = []
        if replace:
            deleted = self.store.delete(source_id=doc.source_id)
            if deleted:
                replaced.append(doc.source_id)

        self._report(on_progress, path, "storing", len(chunks), len(chunks))
        self.store.upsert(chunks)
        self._report(on_progress, path, "done", len(chunks), len(chunks))

        return IngestReport(
            source_ids=[doc.source_id],
            chunks_ingested=len(chunks),
            sources_replaced=replaced,
            errors=[],
        )

    def ingest_many(
        self,
        paths: Iterable[str | Path],
        *,
        metadata: dict[str, Any] | None = None,
        replace: bool = True,
        on_progress: Callable[[IngestProgress], None] | None = None,
        max_workers: int = 4,
    ) -> IngestReport:
        paths = list(paths)
        source_ids: list[str] = []
        replaced: list[str] = []
        errors: list[tuple[str, str]] = []
        chunks_ingested = 0

        def _one(p: str | Path) -> tuple[str, IngestReport | None, str | None]:
            try:
                return (str(p), self.ingest(p, metadata=metadata, replace=replace, on_progress=on_progress), None)
            except Exception as e:  # noqa: BLE001 - collected, not raised, in best-effort batch ingest
                return (str(p), None, str(e))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for path_str, report, error in pool.map(_one, paths):
                if error is not None:
                    errors.append((path_str, error))
                    continue
                source_ids.extend(report.source_ids)
                replaced.extend(report.sources_replaced)
                chunks_ingested += report.chunks_ingested

        return IngestReport(
            source_ids=source_ids,
            chunks_ingested=chunks_ingested,
            sources_replaced=replaced,
            errors=errors,
        )

    def _embed_groups(self, groups: list[ChunkGroup]) -> list[ChunkGroup]:
        if not groups:
            return []
        text_groups = [[c.text for c in g.chunks] for g in groups]
        vector_groups = self.embedder.embed_groups(text_groups, input_type="document")

        out: list[ChunkGroup] = []
        for group, vectors in zip(groups, vector_groups):
            embedded_chunks = [
                dataclasses.replace(chunk, vector=vector)
                for chunk, vector in zip(group.chunks, vectors)
            ]
            out.append(ChunkGroup(group_id=group.group_id, chunks=embedded_chunks))
        return out

    @staticmethod
    def _report(
        on_progress: Callable[[IngestProgress], None] | None,
        path: Path,
        stage: str,
        done: int,
        total: int,
    ) -> None:
        if on_progress is not None:
            on_progress(IngestProgress(path=str(path), stage=stage, chunks_done=done, chunks_total=total))  # type: ignore[arg-type]

    # -- retrieval -------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        k: int = 10,
        filters: Filter | None = None,
        min_score: float | None = None,
    ) -> list[Retrieved]:
        results = self.retriever.retrieve(query, k=k, filters=filters)
        if min_score is not None:
            results = [r for r in results if r.score >= min_score]
        return results

    # -- corpus management -------------------------------------------------

    def get(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return self.store.get(chunk_ids)

    def sources(self) -> list[SourceInfo]:
        return self.store.list_sources()

    def remove(self, source_id: str) -> int:
        return self.store.delete(source_id=source_id)

    def count(self) -> int:
        return self.store.count()
