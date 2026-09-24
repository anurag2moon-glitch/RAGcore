"""build_contextual_embeddings — the construction pass that writes a short
situating blurb for each chunk, embeds `f"{blurb}\\n\\n{chunk.text}"` in
place of the chunk's plain text, and re-upserts the result into the same
VectorStore under the same chunk_id.

A separate checkpointed pass, never inside Corpus.ingest() — mirrors
ragcore.concepts.pipeline.build_concepts for the identical reason: an LLM
call needs its own resumable checkpoint, something Corpus.ingest() has no
mechanism for (see ragcore/concepts/pipeline.py's module docstring for the
full argument, which applies here unchanged). Resume unit is
(source_id, context_group_id) — one chapter, matching the writer's own
per-chapter batching. A chapter's blurbs are written to the ContextStore
(with its checkpoint row, in one transaction) only after the corresponding
chunks have already been embedded and upserted — a crash between the two
merely costs a wasted re-write-blurbs call on retry (VectorStore.upsert is
idempotent by chunk_id), never a correctness gap.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

from ragcore.contextual.base import ContextStore, ContextWriter
from ragcore.graph.pipeline import _group_by_context_group

if TYPE_CHECKING:
    from ragcore.corpus import Corpus

PROMPT_VERSION = "contextual-v1"
SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class ContextProgress:
    source_id: str
    context_group_id: str
    stage: str
    """"writing" | "done" | "skipped" | "failed"."""


@dataclass(frozen=True)
class ContextReport:
    chunks_processed: int
    chapters_processed: int
    chapters_skipped: int
    errors: tuple[tuple[str, str, str], ...]
    """(source_id, context_group_id, error message) for every chapter whose
    blurb-writing raised — collected, not propagated, and retried on the
    next call regardless of `resume` (same convention as
    ragcore.concepts.pipeline.ConceptReport.errors)."""


def _fingerprint(model: str, chunk_ids: Sequence[str]) -> str:
    payload = "|".join([model, PROMPT_VERSION, SCHEMA_VERSION, "\n".join(sorted(chunk_ids))])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_contextual_embeddings(
    corpus: Corpus,
    store: ContextStore,
    writer: ContextWriter,
    *,
    source_ids: Sequence[str] | None = None,
    resume: bool = True,
    on_progress: Callable[[ContextProgress], None] | None = None,
) -> ContextReport:
    sources = corpus.sources()
    if source_ids is not None:
        wanted = set(source_ids)
        sources = [s for s in sources if s.source_id in wanted]

    chunks_processed = 0
    chapters_processed = 0
    chapters_skipped = 0
    errors: list[tuple[str, str, str]] = []

    for source in sources:
        chunks = corpus.store.get_by_source(source.source_id)
        for chapter_chunks in _group_by_context_group(chunks):
            context_group_id = chapter_chunks[0].meta.context_group_id
            chunk_ids = [c.chunk_id for c in chapter_chunks]
            fingerprint = _fingerprint(writer.model, chunk_ids)

            if resume and store.checkpoint(source.source_id, context_group_id) == fingerprint:
                chapters_skipped += 1
                if on_progress:
                    on_progress(ContextProgress(source.source_id, context_group_id, "skipped"))
                continue

            if on_progress:
                on_progress(ContextProgress(source.source_id, context_group_id, "writing"))

            try:
                blurbs = writer.write_blurbs(chapter_chunks)
            except Exception as e:  # noqa: BLE001 - recorded, not raised; one bad chapter shouldn't abort the corpus
                errors.append((source.source_id, context_group_id, str(e)))
                store.mark_chapter_failed(
                    source_id=source.source_id,
                    context_group_id=context_group_id,
                    fingerprint=fingerprint,
                    error=str(e),
                )
                if on_progress:
                    on_progress(ContextProgress(source.source_id, context_group_id, "failed"))
                continue

            chunks_processed += len(chapter_chunks)
            chapters_processed += 1

            # The blurb is prepended only to the string handed to the
            # embedder — chunk.text (and everything downstream that reads
            # it: citations, ClaudeVerifier) stays the original passage.
            contextualized_texts = [
                f"{blurb}\n\n{chunk.text}" if blurb else chunk.text
                for chunk, blurb in zip(chapter_chunks, blurbs, strict=True)
            ]
            embedded = corpus.embedder.embed_groups(
                [contextualized_texts], input_type="document"
            )[0]

            updated_chunks = [
                dataclasses.replace(
                    chunk,
                    vector=vector,
                    meta=dataclasses.replace(
                        chunk.meta, extra={**chunk.meta.extra, "context_blurb": blurb}
                    ),
                )
                for chunk, blurb, vector in zip(chapter_chunks, blurbs, embedded, strict=True)
            ]
            corpus.store.upsert(updated_chunks)

            blurbs_by_id = {
                chunk.chunk_id: blurb
                for chunk, blurb in zip(chapter_chunks, blurbs, strict=True)
            }
            store.add_chapter_blurbs(
                source_id=source.source_id,
                context_group_id=context_group_id,
                fingerprint=fingerprint,
                blurbs=blurbs_by_id,
            )

            if on_progress:
                on_progress(ContextProgress(source.source_id, context_group_id, "done"))

    return ContextReport(
        chunks_processed=chunks_processed,
        chapters_processed=chapters_processed,
        chapters_skipped=chapters_skipped,
        errors=tuple(errors),
    )
