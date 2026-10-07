"""build_hierarchy — the construction pass that builds a chunk -> section ->
chapter -> book summarization hierarchy on top of the existing chunk
structure, using only metadata StructuralChunker already produces
(section_path, chapter_index, section) — no chunker changes needed.

A separate checkpointed pass, run after ingest() — mirrors
ragcore.concepts.pipeline.build_concepts for the same reason every other
LLM-backed pass in this codebase is separate: an LLM call needs its own
resumable checkpoint, which Corpus.ingest() has no mechanism for.

Node ids are positional, not content-addressed like concepts/communities:
`hierarchy:section:{source_id}:{chapter_index}:{section_index}`,
`hierarchy:chapter:{source_id}:{chapter_index}`, `hierarchy:book:{source_id}`.
Unlike a concept or community (which can legitimately recur across
completely different sources and benefits from content-addressed merging),
a hierarchy node's identity is inherently tied to one exact position in one
book's table of contents — a stable positional id is simpler and just as
resumable, since HierarchyStore's per-node checkpoint fingerprint (not the
id) is what actually decides whether a node needs rebuilding.

A chapter with no distinct sub-sections (every chunk sharing one identical
`section_path` — StructuralChunker's own signal for "no real sub-heading")
skips the section level entirely and summarizes its own chunks directly
into the chapter node — call volume scales with real book structure, not an
arbitrary fan-in.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

from ragcore.graph.pipeline import _group_by_context_group
from ragcore.hierarchy.base import HierarchyStore, HierarchySummarizer
from ragcore.hierarchy.types import HierarchyLevel, HierarchyNode

if TYPE_CHECKING:
    from ragcore.corpus import Corpus
    from ragcore.types import Chunk

PROMPT_VERSION = "hierarchy-v1"
SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class HierarchyProgress:
    source_id: str
    node_id: str
    level: str
    stage: str
    """"building" | "done" | "skipped" | "failed"."""


@dataclass(frozen=True)
class HierarchyReport:
    sources_processed: int
    nodes_built: int
    nodes_skipped: int
    errors: tuple[tuple[str, str, str], ...]
    """(source_id, node_id, error message) for every node whose
    summarization raised — collected, not propagated, and retried on the
    next call regardless of `resume` (same convention as
    ragcore.concepts.pipeline.ConceptReport.errors)."""


def _fingerprint(model: str, child_ids: Sequence[str]) -> str:
    payload = "|".join([model, PROMPT_VERSION, SCHEMA_VERSION, "\n".join(sorted(child_ids))])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _group_by_section(chunks: Sequence[Chunk]) -> list[list[Chunk]]:
    groups: dict[tuple, list[Chunk]] = {}
    order: list[tuple] = []
    for c in chunks:
        key = c.meta.section_path
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(c)
    return [groups[key] for key in order]


def build_hierarchy(
    corpus: Corpus,
    store: HierarchyStore,
    summarizer: HierarchySummarizer,
    *,
    source_ids: Sequence[str] | None = None,
    resume: bool = True,
    max_book_batch: int = 20,
    on_progress: Callable[[HierarchyProgress], None] | None = None,
) -> HierarchyReport:
    sources = corpus.sources()
    if source_ids is not None:
        wanted = set(source_ids)
        sources = [s for s in sources if s.source_id in wanted]

    nodes_built = 0
    nodes_skipped = 0
    errors: list[tuple[str, str, str]] = []

    def _build_or_resume(
        node_id: str,
        level: HierarchyLevel,
        source_id: str,
        label: str,
        child_texts: Sequence[str],
        child_ids: Sequence[str],
        source_chunk_ids: Sequence[str],
        tenant_id: str | None,
    ) -> HierarchyNode | None:
        nonlocal nodes_built, nodes_skipped
        fingerprint = _fingerprint(summarizer.model, child_ids)

        if resume and store.checkpoint(node_id) == fingerprint:
            nodes_skipped += 1
            if on_progress:
                on_progress(HierarchyProgress(source_id, node_id, level, "skipped"))
            return store.node(node_id)

        if on_progress:
            on_progress(HierarchyProgress(source_id, node_id, level, "building"))

        try:
            # The LLM call always runs before any transaction opens (inside
            # summarizer.summarize / store.add_node) — same discipline as
            # every other checkpointed pass in this codebase.
            node = summarizer.summarize(
                node_id=node_id,
                level=level,
                source_id=source_id,
                label=label,
                child_texts=child_texts,
                child_ids=child_ids,
                source_chunk_ids=source_chunk_ids,
            )
        except Exception as e:  # noqa: BLE001 - recorded, not raised; one bad node shouldn't abort the corpus
            errors.append((source_id, node_id, str(e)))
            store.mark_node_failed(node_id=node_id, fingerprint=fingerprint, error=str(e))
            if on_progress:
                on_progress(HierarchyProgress(source_id, node_id, level, "failed"))
            return None

        vector = corpus.embedder.embed_groups([[node.summary]], input_type="document")[0][0]
        node = dataclasses.replace(node, vector=tuple(vector), tenant_id=tenant_id)
        store.add_node(node, fingerprint=fingerprint)
        nodes_built += 1
        if on_progress:
            on_progress(HierarchyProgress(source_id, node_id, level, "done"))
        return node

    for source in sources:
        chunks = corpus.store.get_by_source(source.source_id)
        source_tenant_id = chunks[0].meta.tenant_id if chunks else None
        chapter_node_ids: list[str] = []
        chapter_texts: list[str] = []

        for chapter_chunks in _group_by_context_group(chunks):
            chapter_index = chapter_chunks[0].meta.chapter_index
            chapter_title = chapter_chunks[0].meta.chapter_title or f"Chapter {chapter_index}"
            section_groups = _group_by_section(chapter_chunks)

            if len(section_groups) > 1:
                section_node_ids: list[str] = []
                section_texts: list[str] = []
                for section_index, section_chunks in enumerate(section_groups):
                    section_label = section_chunks[0].meta.section or chapter_title
                    section_chunk_ids = [c.chunk_id for c in section_chunks]
                    section_node_id = (
                        f"hierarchy:section:{source.source_id}:{chapter_index}:{section_index}"
                    )
                    node = _build_or_resume(
                        section_node_id,
                        "section",
                        source.source_id,
                        section_label,
                        [c.text for c in section_chunks],
                        section_chunk_ids,
                        section_chunk_ids,
                        source_tenant_id,
                    )
                    if node is None:
                        continue  # this section failed — skip it, the chapter proceeds without it
                    section_node_ids.append(node.node_id)
                    section_texts.append(node.summary)
                chapter_child_ids = section_node_ids
                chapter_child_texts = section_texts
                chapter_source_chunk_ids = [c.chunk_id for c in chapter_chunks]
            else:
                chapter_child_ids = [c.chunk_id for c in chapter_chunks]
                chapter_child_texts = [c.text for c in chapter_chunks]
                chapter_source_chunk_ids = chapter_child_ids

            if not chapter_child_ids:
                continue  # every section failed — nothing to summarize this chapter from

            chapter_node_id = f"hierarchy:chapter:{source.source_id}:{chapter_index}"
            chapter_node = _build_or_resume(
                chapter_node_id,
                "chapter",
                source.source_id,
                chapter_title,
                chapter_child_texts,
                chapter_child_ids,
                chapter_source_chunk_ids,
                source_tenant_id,
            )
            if chapter_node is None:
                continue
            chapter_node_ids.append(chapter_node.node_id)
            chapter_texts.append(chapter_node.summary)

        if not chapter_node_ids:
            continue

        book_child_ids = chapter_node_ids
        book_child_texts = chapter_texts
        if len(chapter_texts) > max_book_batch:
            # Map-reduce for a book with more chapters than fit in one
            # prompt — an ephemeral intermediate pass, never persisted as
            # its own HierarchyNode (only section/chapter/book are stored).
            try:
                batch_texts = []
                for i in range(0, len(chapter_texts), max_book_batch):
                    batch = chapter_texts[i : i + max_book_batch]
                    batch_ids = chapter_node_ids[i : i + max_book_batch]
                    intermediate = summarizer.summarize(
                        node_id=f"hierarchy:book-batch:{source.source_id}:{i}",
                        level="book",
                        source_id=source.source_id,
                        label=source.title or source.path,
                        child_texts=batch,
                        child_ids=batch_ids,
                        source_chunk_ids=(),
                    )
                    batch_texts.append(intermediate.summary)
                book_child_texts = batch_texts
            except Exception as e:  # noqa: BLE001 - recorded; book-level summary skipped for this source
                errors.append((source.source_id, f"hierarchy:book:{source.source_id}", str(e)))
                continue

        book_node_id = f"hierarchy:book:{source.source_id}"
        _build_or_resume(
            book_node_id,
            "book",
            source.source_id,
            source.title or source.path,
            book_child_texts,
            book_child_ids,
            [c.chunk_id for c in chunks],
            source_tenant_id,
        )

    return HierarchyReport(
        sources_processed=len(sources),
        nodes_built=nodes_built,
        nodes_skipped=nodes_skipped,
        errors=tuple(errors),
    )
