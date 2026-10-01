"""Corpus — the single object consuming projects hold.

Orchestration only: ingestion wires Loader -> Chunker -> Embedder -> Store,
retrieval delegates to a Retriever. No I/O logic of its own beyond calling
into those components, so swapping any one of them never touches this file.
"""

from __future__ import annotations

import dataclasses
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ragcore.chunking.base import Chunker
from ragcore.chunking.structural import StructuralChunker
from ragcore.embedding.base import Embedder
from ragcore.embedding.cache import CacheKey, EmbeddingCache
from ragcore.errors import RagcoreError
from ragcore.generation.base import Answer, Generator
from ragcore.ingestion.base import IngestCheckpointStore
from ragcore.ingestion.pipeline import group_fingerprint
from ragcore.loaders import LoaderRegistry
from ragcore.retrieval.base import Retriever
from ragcore.retrieval.vector import VectorRetriever
from ragcore.stores.base import VectorStore
from ragcore.types import (
    Chunk,
    ChunkGroup,
    Document,
    Filter,
    IngestProgress,
    IngestReport,
    IngestStage,
    Retrieved,
    SourceInfo,
)


class Corpus:
    """A book collection: ingest files into it, retrieve chunks from it.

    `retriever` is a public, reassignable attribute — this is the seam the
    future graph layer uses. `Corpus.retriever = GraphRetriever(corpus.retriever,
    graph_store, corpus)` swaps in graph-aware retrieval without changing any
    `corpus.retrieve(...)` call site.

    `generator` is similarly public and optional (default None): retrieval
    works with no generator configured at all, and `ask()` only requires one
    be set when it's actually called.

    `checkpoint_store` is optional (default None) and opt-in: see
    `ragcore.ingestion` and docs/adr/0002-ingest-checkpointing-opt-in.md.
    Leaving it unset keeps `ingest()`'s original single-shot behavior
    (embed everything, then one final `store.upsert()`) with zero new
    dependency; setting it makes `ingest()` resumable per-ChunkGroup, at
    the cost of upserting each group as soon as it's embedded instead of
    only once at the end.

    `embedding_cache` is optional (default None): see
    `ragcore.embedding.cache`. When set, `_embed_groups` skips the
    Embedder call entirely for any ChunkGroup whose every chunk's
    (content_hash, model_name, dim) is already cached — e.g. re-ingesting
    an unchanged file, or a second book sharing boilerplate text with the
    first, costs no embedding calls for that content the second time.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        chunker: Chunker | None = None,
        loaders: LoaderRegistry | None = None,
        generator: Generator | None = None,
        checkpoint_store: IngestCheckpointStore | None = None,
        embedding_cache: EmbeddingCache | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.chunker = chunker or StructuralChunker()
        self.loaders = loaders or LoaderRegistry()
        self.retriever: Retriever = VectorRetriever(embedder, store)
        self.generator = generator
        self.checkpoint_store = checkpoint_store
        self.embedding_cache = embedding_cache

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

        if metadata:
            groups = [
                ChunkGroup(
                    group_id=g.group_id,
                    chunks=[
                        dataclasses.replace(
                            c,
                            meta=dataclasses.replace(
                                c.meta, extra={**c.meta.extra, **metadata}
                            ),
                        )
                        for c in g.chunks
                    ],
                )
                for g in groups
            ]

        if self.checkpoint_store is not None:
            return self._ingest_checkpointed(
                path,
                doc,
                groups,
                total_chunks,
                metadata=metadata,
                replace=replace,
                on_progress=on_progress,
            )

        self._report(on_progress, path, "embedding", 0, total_chunks)
        embedded_groups = self._embed_groups(groups)
        chunks = [chunk for group in embedded_groups for chunk in group.chunks]

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

    def _ingest_checkpointed(
        self,
        path: Path,
        doc: Document,
        groups: list[ChunkGroup],
        total_chunks: int,
        *,
        metadata: dict[str, Any] | None,
        replace: bool,
        on_progress: Callable[[IngestProgress], None] | None,
    ) -> IngestReport:
        """ingest()'s path when a checkpoint_store is configured: embeds and
        upserts one ChunkGroup at a time, checkpointing each immediately
        after its upsert succeeds, so a retried ingest() of the same file
        resumes rather than restarting — see
        docs/adr/0002-ingest-checkpointing-opt-in.md."""
        checkpoint_store = self.checkpoint_store
        assert checkpoint_store is not None

        fingerprints = {
            g.group_id: group_fingerprint(
                embedder_name=self.embedder.name, metadata=metadata, chunks=g.chunks
            )
            for g in groups
        }
        resumed_group_ids = {
            g.group_id
            for g in groups
            if checkpoint_store.checkpoint(doc.source_id, g.group_id)
            == fingerprints[g.group_id]
        }

        replaced: list[str] = []
        if replace:
            resumed_chunk_ids = {
                c.chunk_id
                for g in groups
                if g.group_id in resumed_group_ids
                for c in g.chunks
            }
            existing = self.store.get_by_source(doc.source_id)
            to_delete = [c.chunk_id for c in existing if c.chunk_id not in resumed_chunk_ids]
            if to_delete:
                deleted = self.store.delete(chunk_ids=to_delete)
                if deleted:
                    replaced.append(doc.source_id)

        chunks_done = sum(len(g.chunks) for g in groups if g.group_id in resumed_group_ids)
        for g in groups:
            if g.group_id in resumed_group_ids:
                continue

            fingerprint = fingerprints[g.group_id]
            self._report(on_progress, path, "embedding", chunks_done, total_chunks)
            try:
                embedded_group = self._embed_groups([g])[0]
            except Exception as e:  # noqa: BLE001 - checkpointed as failed, then re-raised unchanged
                checkpoint_store.mark_group_failed(
                    source_id=doc.source_id,
                    group_id=g.group_id,
                    fingerprint=fingerprint,
                    error=str(e),
                )
                raise

            self.store.upsert(embedded_group.chunks)
            checkpoint_store.mark_group_done(
                source_id=doc.source_id, group_id=g.group_id, fingerprint=fingerprint
            )
            chunks_done += len(embedded_group.chunks)
            self._report(on_progress, path, "storing", chunks_done, total_chunks)

        self._report(on_progress, path, "done", chunks_done, total_chunks)
        return IngestReport(
            source_ids=[doc.source_id],
            chunks_ingested=chunks_done,
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
        max_retries: int = 0,
    ) -> IngestReport:
        """`max_retries` (default 0, today's single-attempt behavior): with
        a `checkpoint_store` configured, a failing file is retried in place
        up to `max_retries` additional times — each retry resumes from its
        own last-completed ChunkGroup rather than restarting. After
        exhausting retries, the failure is recorded via
        `checkpoint_store.mark_dead_letter()` in addition to the returned
        `errors` list, so it survives past this one call. Ignored when no
        `checkpoint_store` is configured beyond simply not being retried
        (set `max_retries=0`, the default, to skip retrying entirely)."""
        paths = list(paths)
        source_ids: list[str] = []
        replaced: list[str] = []
        errors: list[tuple[str, str]] = []
        chunks_ingested = 0

        def _one(p: str | Path) -> tuple[str, IngestReport | None, str | None]:
            attempts = 0
            while True:
                attempts += 1
                try:
                    report = self.ingest(p, metadata=metadata, replace=replace, on_progress=on_progress)
                except Exception as e:  # noqa: BLE001 - collected, not raised, in best-effort batch ingest
                    if attempts <= max_retries:
                        continue
                    if self.checkpoint_store is not None:
                        self.checkpoint_store.mark_dead_letter(
                            path=str(p), error=str(e), attempts=attempts
                        )
                    return (str(p), None, str(e))
                else:
                    if self.checkpoint_store is not None:
                        self.checkpoint_store.clear_dead_letter(str(p))
                    return (str(p), report, None)

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
        if self.embedding_cache is not None:
            return self._embed_groups_with_cache(groups)

        text_groups = [[c.text for c in g.chunks] for g in groups]
        vector_groups = self.embedder.embed_groups(text_groups, input_type="document")
        return self._apply_vectors(groups, vector_groups)

    def _embed_groups_with_cache(self, groups: list[ChunkGroup]) -> list[ChunkGroup]:
        """Caches at ChunkGroup granularity, not per-chunk: a contextualized
        Embedder (e.g. VoyageContextEmbedder) embeds every chunk in a group
        aware of its neighbors, so sending only a miss-subset of a group
        would change what the embedder sees and risk a different vector
        than if the whole group had been embedded together. A group is
        only skipped when *every* chunk in it is already cached; otherwise
        the whole group is re-embedded (and every chunk in it re-cached),
        same as if there were no cache for that group at all."""
        cache = self.embedding_cache
        assert cache is not None
        model_name, dim = self.embedder.name, self.embedder.dim

        group_keys = [
            [self._cache_key(c.text, model_name, dim) for c in g.chunks] for g in groups
        ]
        all_keys = [key for keys in group_keys for key in keys]
        cached = cache.get_many(all_keys) if all_keys else {}

        miss_groups = [
            g for g, keys in zip(groups, group_keys, strict=True)
            if any(key not in cached for key in keys)
        ]
        if miss_groups:
            text_groups = [[c.text for c in g.chunks] for g in miss_groups]
            freshly_embedded = self.embedder.embed_groups(text_groups, input_type="document")
            new_entries: dict[CacheKey, Any] = {}
            for g, vectors in zip(miss_groups, freshly_embedded, strict=True):
                for c, v in zip(g.chunks, vectors, strict=True):
                    new_entries[self._cache_key(c.text, model_name, dim)] = v
            cache.put_many(new_entries)
            cached = {**cached, **new_entries}

        out: list[ChunkGroup] = []
        for group, keys in zip(groups, group_keys, strict=True):
            vectors = [cached[key] for key in keys]
            out.append(ChunkGroup(group_id=group.group_id, chunks=[
                dataclasses.replace(c, vector=v)
                for c, v in zip(group.chunks, vectors, strict=True)
            ]))
        return out

    @staticmethod
    def _apply_vectors(groups: list[ChunkGroup], vector_groups: list[list[Any]]) -> list[ChunkGroup]:
        out: list[ChunkGroup] = []
        for group, vectors in zip(groups, vector_groups, strict=True):
            embedded_chunks = [
                dataclasses.replace(chunk, vector=vector)
                for chunk, vector in zip(group.chunks, vectors, strict=True)
            ]
            out.append(ChunkGroup(group_id=group.group_id, chunks=embedded_chunks))
        return out

    @staticmethod
    def _cache_key(text: str, model_name: str, dim: int) -> CacheKey:
        return (hashlib.sha256(text.encode("utf-8")).hexdigest(), model_name, dim)

    @staticmethod
    def _report(
        on_progress: Callable[[IngestProgress], None] | None,
        path: Path,
        stage: IngestStage,
        done: int,
        total: int,
    ) -> None:
        if on_progress is not None:
            on_progress(IngestProgress(path=str(path), stage=stage, chunks_done=done, chunks_total=total))

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

    def ask(
        self,
        query: str,
        *,
        k: int = 10,
        filters: Filter | None = None,
        generator: Generator | None = None,
    ) -> Answer:
        """Retrieve then synthesize: `retrieve(query, ...)` followed by a
        Generator call over the results. Additive sugar over `retrieve()` —
        it changes nothing about how retrieval itself works, and a Corpus
        that never calls `ask()` never needs a Generator configured at all.
        """
        gen = generator or self.generator
        if gen is None:
            raise RagcoreError(
                "no Generator configured; pass generator=... to ask() or set corpus.generator"
            )
        results = self.retrieve(query, k=k, filters=filters)
        return gen.generate(query, results)

    # -- corpus management -------------------------------------------------

    def get(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return self.store.get(chunk_ids)

    def sources(self) -> list[SourceInfo]:
        return self.store.list_sources()

    def remove(self, source_id: str) -> int:
        return self.store.delete(source_id=source_id)

    def count(self) -> int:
        return self.store.count()
