"""build_concepts — construction pass building the concept/theme graph from
already-ingested chunks.

Mirrors ragcore.graph.pipeline.build_graph closely (same chapter-grouping
via _group_by_context_group, the same content-addressed-id-collapses-
exact-matches-for-free shortcut, the same resolver-merge id-remap pattern)
but adds the one thing build_graph doesn't need: a per-chapter checkpoint. A
concept corpus can be far larger than a single-book entity extraction run,
and losing partial progress on a multi-hour ingestion is a real cost
build_graph never had to plan for.

Resumability contract (see also ragcore/concepts/sqlite_store.py and
CLAUDE.md): the resume unit is (source_id, context_group_id) — one chapter,
matching the extraction call granularity. A chapter is skipped when
ConceptStore.checkpoint() already holds a fingerprint matching this run's
(sha256 of extractor.model, the prompt/schema version, and the chapter's own
sorted chunk_ids) — so a model change, a prompt-version bump, or a chunker
change that shifts chunk boundaries all invalidate it automatically, the
last one for free since chunk_ids are themselves content-addressed. A
chapter's concepts, edges, and checkpoint row are written in one transaction
(ConceptStore.add_chapter_concepts) so a crash never leaves a chapter
half-written; extraction itself always runs before that transaction opens,
never inside it, so a slow LLM call never holds SQLite's write lock.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

from ragcore.concepts.base import ConceptExtractor, ConceptStore
from ragcore.concepts.resolver import ConceptResolver, EmbeddingConceptResolver, concept_text
from ragcore.concepts.types import ConceptEdge
from ragcore.graph.pipeline import _group_by_context_group

logger = logging.getLogger("ragcore.concepts")

if TYPE_CHECKING:
    from ragcore.corpus import Corpus

PROMPT_VERSION = "concepts-v1"
SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class ConceptProgress:
    source_id: str
    context_group_id: str
    stage: str
    """"extracting" | "done" | "skipped" | "failed"."""


@dataclass(frozen=True)
class ConceptReport:
    chunks_processed: int
    chapters_processed: int
    chapters_skipped: int
    concepts_extracted: int
    edges_extracted: int
    concepts_merged: int
    errors: tuple[tuple[str, str, str], ...]
    """(source_id, context_group_id, error message) for every chapter whose
    extraction raised — collected, not propagated, so one bad chapter
    doesn't abort the whole corpus. Each is also recorded in the
    ConceptStore itself (status='failed') and will be retried on the next
    build_concepts() call regardless of `resume`."""


def _fingerprint(model: str, chunk_ids: Sequence[str]) -> str:
    payload = "|".join([model, PROMPT_VERSION, SCHEMA_VERSION, "\n".join(sorted(chunk_ids))])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _remap_edge(edge: ConceptEdge, id_remap: dict[str, str]) -> ConceptEdge:
    # Local import: ragcore.concepts.claude_extractor is the one module that
    # owns the edge-id canonicalization rule (sorted endpoints); reuse it
    # rather than re-deriving "sort, then f-string" here too.
    from ragcore.concepts.claude_extractor import _edge_id

    source = id_remap.get(edge.source_concept_id, edge.source_concept_id)
    target = id_remap.get(edge.target_concept_id, edge.target_concept_id)
    edge_id, canonical_source, canonical_target = _edge_id(source, target)
    return dataclasses.replace(
        edge, edge_id=edge_id, source_concept_id=canonical_source, target_concept_id=canonical_target
    )


def build_concepts(
    corpus: Corpus,
    store: ConceptStore,
    extractor: ConceptExtractor,
    resolver: ConceptResolver | None = None,
    *,
    source_ids: Sequence[str] | None = None,
    resume: bool = True,
    on_progress: Callable[[ConceptProgress], None] | None = None,
) -> ConceptReport:
    resolver = resolver or EmbeddingConceptResolver(corpus.embedder)

    sources = corpus.sources()
    if source_ids is not None:
        wanted = set(source_ids)
        sources = [s for s in sources if s.source_id in wanted]

    chunks_processed = 0
    chapters_processed = 0
    chapters_skipped = 0
    concepts_extracted = 0
    edges_extracted = 0
    concepts_merged = 0
    errors: list[tuple[str, str, str]] = []

    for source in sources:
        chunks = corpus.store.get_by_source(source.source_id)
        for chapter_chunks in _group_by_context_group(chunks):
            context_group_id = chapter_chunks[0].meta.context_group_id
            chunk_ids = [c.chunk_id for c in chapter_chunks]
            fingerprint = _fingerprint(extractor.model, chunk_ids)

            if resume and store.checkpoint(source.source_id, context_group_id) == fingerprint:
                chapters_skipped += 1
                if on_progress:
                    on_progress(ConceptProgress(source.source_id, context_group_id, "skipped"))
                continue

            if on_progress:
                on_progress(ConceptProgress(source.source_id, context_group_id, "extracting"))

            try:
                # Extraction always runs before any transaction opens — see
                # module docstring for why holding BEGIN IMMEDIATE across an
                # API call would be a real concurrency hazard.
                candidate_concepts, candidate_edges = extractor.extract(chapter_chunks)
            except Exception as e:  # noqa: BLE001 - recorded, not raised; one bad chapter shouldn't abort the corpus
                logger.warning("build_concepts: %s extraction failed: %s", context_group_id, e)
                errors.append((source.source_id, context_group_id, str(e)))
                store.mark_chapter_failed(
                    source_id=source.source_id,
                    context_group_id=context_group_id,
                    fingerprint=fingerprint,
                    error=str(e),
                )
                if on_progress:
                    on_progress(ConceptProgress(source.source_id, context_group_id, "failed"))
                continue

            chunks_processed += len(chapter_chunks)
            chapters_processed += 1

            # store.all_concepts() is fetched once per chapter, not once per
            # candidate: nothing is written back to `store` until this whole
            # chapter's candidates are resolved, so the comparison set is
            # identical for every candidate in this call — the same
            # per-chapter (not per-candidate) caching rationale
            # ragcore.graph.pipeline.build_graph already uses for
            # entities_by_type, just without a type dimension to key on.
            existing_concepts = store.all_concepts()
            existing_ids = {c.concept_id for c in existing_concepts}

            id_remap: dict[str, str] = {}
            new_ids: list[str] = []
            for candidate in candidate_concepts:
                if candidate.concept_id in existing_ids:
                    # Exact content-addressed match — add_chapter_concepts()
                    # merges provenance naturally; no resolver call, no
                    # embedding spent on an unambiguous case.
                    final_id = candidate.concept_id
                else:
                    merge_into = resolver.resolve(candidate, existing_concepts)
                    final_id = merge_into or candidate.concept_id
                    if merge_into is not None:
                        concepts_merged += 1
                    else:
                        new_ids.append(candidate.concept_id)
                id_remap[candidate.concept_id] = final_id

            # Batch-embed only the genuinely new concepts: not an exact
            # match, not a resolver merge. An existing concept_id already
            # has a stored vector, and add_chapter_concepts never clobbers
            # it with None — see Concept.vector's docstring.
            vectors: dict[str, tuple[float, ...]] = {}
            if new_ids:
                by_id = {c.concept_id: c for c in candidate_concepts}
                texts = [concept_text(by_id[cid]) for cid in new_ids]
                embedded = corpus.embedder.embed_groups([[t] for t in texts], input_type="document")
                vectors = {
                    cid: tuple(vecs[0]) for cid, vecs in zip(new_ids, embedded, strict=True)
                }

            resolved_concepts = [
                dataclasses.replace(
                    candidate,
                    concept_id=id_remap[candidate.concept_id],
                    vector=vectors.get(id_remap[candidate.concept_id]),
                )
                for candidate in candidate_concepts
            ]
            remapped_edges = [_remap_edge(edge, id_remap) for edge in candidate_edges]

            store.add_chapter_concepts(
                source_id=source.source_id,
                context_group_id=context_group_id,
                fingerprint=fingerprint,
                concepts=resolved_concepts,
                edges=remapped_edges,
            )

            concepts_extracted += len(resolved_concepts)
            edges_extracted += len(remapped_edges)
            logger.debug(
                "build_concepts: %s -> %d concepts, %d edges",
                context_group_id, len(resolved_concepts), len(remapped_edges),
            )
            if on_progress:
                on_progress(ConceptProgress(source.source_id, context_group_id, "done"))

    logger.info(
        "build_concepts: done — %d/%d chapters processed (%d skipped, %d failed), "
        "%d concepts (%d merged), %d edges",
        chapters_processed, chapters_processed + chapters_skipped + len(errors), chapters_skipped,
        len(errors), concepts_extracted, concepts_merged, edges_extracted,
    )
    return ConceptReport(
        chunks_processed=chunks_processed,
        chapters_processed=chapters_processed,
        chapters_skipped=chapters_skipped,
        concepts_extracted=concepts_extracted,
        edges_extracted=edges_extracted,
        concepts_merged=concepts_merged,
        errors=tuple(errors),
    )
