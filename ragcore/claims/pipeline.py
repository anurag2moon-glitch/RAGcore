"""build_claims — construction pass extracting SourceClaims from
already-ingested chunks, anchored to the concept layer's already-extracted
concepts.

Mirrors ragcore.concepts.pipeline.build_concepts closely: the same
chapter-grouping via _group_by_context_group, the same
LLM-call-before-transaction-opens discipline, the same per-chapter
checkpoint. Requires a ConceptStore — build_concepts() must have already run
over a chapter before build_claims() can anchor claims to its concepts (see
ragcore.versioning.sync.sync_corpus, which now runs build_claims
immediately after build_concepts for exactly this reason).

Resumability contract: the resume unit is (source_id, context_group_id),
matching build_concepts. A chapter is skipped when ClaimStore.checkpoint()
already holds a fingerprint matching this run's
(model, prompt_version, schema_version, sorted chunk_ids, sorted concept_ids)
— the concept_ids term is required, not optional: build_concepts() remaps
concept ids through its resolver, so a chapter's concept set can shift
between runs even when its own chunks haven't. Without this term a claim
would stay checkpointed 'done' pointing at concept ids that may no longer
exist.

A chapter whose concept menu is empty (concepts layer not yet built for it,
or its concept extraction previously failed) is skipped WITHOUT writing any
checkpoint row at all — an unanchored claim can never be linked, and leaving
no checkpoint means a later build_claims() run (once build_concepts() has
caught up) picks the chapter up for free rather than being permanently
skipped.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

from ragcore.claims.base import ClaimExtractor, ClaimStore
from ragcore.graph.pipeline import _group_by_context_group

if TYPE_CHECKING:
    from ragcore.concepts.base import ConceptStore
    from ragcore.corpus import Corpus

PROMPT_VERSION = "claims-v1"
SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class ClaimProgress:
    source_id: str
    context_group_id: str
    stage: str
    """"extracting" | "done" | "skipped" | "skipped_no_concepts" | "failed"."""


@dataclass(frozen=True)
class ClaimReport:
    chunks_processed: int
    chapters_processed: int
    chapters_skipped: int
    chapters_skipped_no_concepts: int
    """Chapters skipped because they had no concepts to anchor claims to yet
    — no checkpoint was written for these, so they'll be retried on the next
    build_claims() call regardless of `resume`."""

    claims_extracted: int
    errors: tuple[tuple[str, str, str], ...]
    """(source_id, context_group_id, error message) for every chapter whose
    extraction raised — collected, not propagated, so one bad chapter
    doesn't abort the whole corpus."""


def _fingerprint(model: str, chunk_ids: Sequence[str], concept_ids: Sequence[str]) -> str:
    payload = "|".join([
        model, PROMPT_VERSION, SCHEMA_VERSION,
        "\n".join(sorted(chunk_ids)), "\n".join(sorted(concept_ids)),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_claims(
    corpus: Corpus,
    store: ClaimStore,
    extractor: ClaimExtractor,
    concept_store: ConceptStore,
    *,
    source_ids: Sequence[str] | None = None,
    resume: bool = True,
    on_progress: Callable[[ClaimProgress], None] | None = None,
) -> ClaimReport:
    sources = corpus.sources()
    if source_ids is not None:
        wanted = set(source_ids)
        sources = [s for s in sources if s.source_id in wanted]

    chunks_processed = 0
    chapters_processed = 0
    chapters_skipped = 0
    chapters_skipped_no_concepts = 0
    claims_extracted = 0
    errors: list[tuple[str, str, str]] = []

    for source in sources:
        chunks = corpus.store.get_by_source(source.source_id)
        for chapter_chunks in _group_by_context_group(chunks):
            context_group_id = chapter_chunks[0].meta.context_group_id
            chunk_ids = [c.chunk_id for c in chapter_chunks]

            concepts_by_chunk = concept_store.concepts_for_chunks(chunk_ids)
            concepts_by_id = {
                c.concept_id: c for concepts in concepts_by_chunk.values() for c in concepts
            }
            chapter_concepts = sorted(concepts_by_id.values(), key=lambda c: c.concept_id)

            if not chapter_concepts:
                chapters_skipped_no_concepts += 1
                if on_progress:
                    on_progress(ClaimProgress(source.source_id, context_group_id, "skipped_no_concepts"))
                continue

            concept_ids = [c.concept_id for c in chapter_concepts]
            fingerprint = _fingerprint(extractor.model, chunk_ids, concept_ids)

            if resume and store.checkpoint(source.source_id, context_group_id) == fingerprint:
                chapters_skipped += 1
                if on_progress:
                    on_progress(ClaimProgress(source.source_id, context_group_id, "skipped"))
                continue

            if on_progress:
                on_progress(ClaimProgress(source.source_id, context_group_id, "extracting"))

            try:
                # Extraction always runs before any transaction opens — same
                # discipline as build_concepts.
                candidate_claims = extractor.extract(chunks=chapter_chunks, concepts=chapter_concepts)
            except Exception as e:  # noqa: BLE001 - recorded, not raised; one bad chapter shouldn't abort the corpus
                errors.append((source.source_id, context_group_id, str(e)))
                store.mark_chapter_failed(
                    source_id=source.source_id, context_group_id=context_group_id,
                    fingerprint=fingerprint, error=str(e),
                )
                if on_progress:
                    on_progress(ClaimProgress(source.source_id, context_group_id, "failed"))
                continue

            chunks_processed += len(chapter_chunks)
            chapters_processed += 1

            # Embed only genuinely-new claim texts: content-addressing
            # already makes a claim re-extracted identically a no-op merge,
            # so there's no reason to re-embed something already stored.
            new_claims = [c for c in candidate_claims if store.claim(c.claim_id) is None]
            if new_claims:
                texts = [c.text for c in new_claims]
                embedded = corpus.embedder.embed_groups([[t] for t in texts], input_type="document")
                vectors = {
                    c.claim_id: tuple(vecs[0])
                    for c, vecs in zip(new_claims, embedded, strict=True)
                }
                candidate_claims = [
                    dataclasses.replace(c, vector=vectors[c.claim_id]) if c.claim_id in vectors else c
                    for c in candidate_claims
                ]

            store.add_chapter_claims(
                source_id=source.source_id, context_group_id=context_group_id,
                fingerprint=fingerprint, claims=candidate_claims,
            )

            claims_extracted += len(candidate_claims)
            if on_progress:
                on_progress(ClaimProgress(source.source_id, context_group_id, "done"))

    return ClaimReport(
        chunks_processed=chunks_processed,
        chapters_processed=chapters_processed,
        chapters_skipped=chapters_skipped,
        chapters_skipped_no_concepts=chapters_skipped_no_concepts,
        claims_extracted=claims_extracted,
        errors=tuple(errors),
    )
