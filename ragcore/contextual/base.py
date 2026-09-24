"""ContextWriter protocol: generates a short situating blurb for each chunk
in a chapter, given the whole chapter as context — the same chapter-batching
rationale ragcore.graph.claude_extractor and ragcore.concepts.claude_extractor
already use for pronoun resolution and consistent labeling.

Nothing in ragcore.types references ragcore.contextual, so — like
ragcore.concepts — this package has no import cycle to defer around.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ragcore.types import Chunk


@runtime_checkable
class ContextWriter(Protocol):
    model: str

    def write_blurbs(self, chunks: Sequence[Chunk]) -> list[str]:
        """One 1-2 sentence blurb per input chunk, in order — what section
        or chapter this chunk is from and what it's discussing in relation
        to surrounding content. Only ever prepended to the *text handed to
        the embedder*, never to Chunk.text itself (see
        ragcore.contextual.pipeline.build_contextual_embeddings) — a blurb
        must never leak into what's shown to a user as "the source
        passage," including in citations and groundedness verification."""
        ...


@runtime_checkable
class ContextStore(Protocol):
    """Persists chunk_id -> blurb text and tracks per-chapter checkpoints
    for build_contextual_embeddings()'s resumability — same shape as
    ragcore.concepts.base.ConceptStore's checkpoint methods."""

    def add_chapter_blurbs(
        self, *, source_id: str, context_group_id: str, fingerprint: str, blurbs: dict[str, str]
    ) -> None: ...

    def checkpoint(self, source_id: str, context_group_id: str) -> str | None: ...

    def mark_chapter_failed(
        self, *, source_id: str, context_group_id: str, fingerprint: str, error: str
    ) -> None: ...

    def blurb(self, chunk_id: str) -> str | None: ...

    def remove_source(self, source_id: str) -> None: ...
