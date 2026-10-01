"""StructuralChunker — the default Chunker.

Groups a Document's blocks by chapter (the first element of section_path),
preserving document order, so that each ChunkGroup is one contextualization
window handed to Embedder.embed_groups() as a unit (see ragcore.embedding).
A chapter whose content exceeds `max_group_tokens` is split into multiple
sequential groups, since a contextualized-embedding request has a hard cap
on total tokens per group (voyage-context-4: 32K without auto-chunking —
`max_group_tokens` defaults to 30_000 to leave headroom).

Within a group, chunks are assembled by greedily packing paragraphs up to
`chunk_size_tokens`, with `chunk_overlap_tokens` of trailing context carried
into the next chunk. A paragraph larger than the budget by itself is
further split by sentence, then (if a single sentence is still too large)
by a hard token slice — this always terminates.

Offset convention: `Chunk.meta.char_start`/`char_end` are offsets into the
group's own concatenated source text (each block's text joined by "\\n\\n",
in the order the blocks appear in that group) — not the whole document.
This keeps offsets well-defined for a chunk that spans multiple blocks
without requiring the full document text to be held in memory at once.

A chunk whose units all come from one page-bearing Block (PdfLoader only
— see ragcore.types.Block.page_hash) gets `ChunkMeta.extra["page_hash"]`
set to that page's content hash, for page-level dedupe; a chunk that
straddles a page boundary gets no page_hash at all, deliberately, rather
than one page's hash standing in for a mix of two.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragcore.chunking.tokenizer import count_tokens
from ragcore.types import Block, Chunk, ChunkGroup, ChunkMeta, Document

_BLANK_LINE = re.compile(r"\n[ \t\r]*\n")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class _Unit:
    text: str
    page: int | None
    page_hash: str | None
    section_path: tuple[str, ...]
    start: int
    end: int
    tokens: int


def _split_paragraphs(text: str) -> list[tuple[str, int, int]]:
    """Split `text` into (stripped_paragraph, start, end) on blank-line
    boundaries. start/end are exact offsets into `text`."""
    results: list[tuple[str, int, int]] = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        start = i
        m = _BLANK_LINE.search(text, i)
        end = m.start() if m else n
        para = text[start:end]
        stripped = para.strip()
        if stripped:
            lstrip_len = len(para) - len(para.lstrip())
            real_start = start + lstrip_len
            real_end = real_start + len(stripped)
            results.append((stripped, real_start, real_end))
        i = end
    return results


def _split_oversized(text: str, start: int, budget_tokens: int) -> list[tuple[str, int, int]]:
    """Split a too-large paragraph into pieces under `budget_tokens`, first
    by sentence boundary, then (only if a single sentence is still too
    large) by a hard character slice sized to the budget. Always
    terminates: the character-slice branch shrinks the remainder every
    iteration regardless of token density.
    """
    if count_tokens(text) <= budget_tokens:
        return [(text, start, start + len(text))]

    pieces: list[tuple[str, int, int]] = []
    pos = 0
    for sent in _SENTENCE_BOUNDARY.split(text):
        if not sent:
            continue
        local_start = text.index(sent, pos)
        pos = local_start + len(sent)
        if count_tokens(sent) <= budget_tokens:
            pieces.append((sent, start + local_start, start + local_start + len(sent)))
            continue
        # Even one sentence is too large — hard-slice by characters. A
        # char-per-token ratio of 3 undershoots typical English tokenization,
        # guaranteeing each slice is under budget without re-measuring in a loop.
        approx_chars = max(1, budget_tokens * 3)
        i = 0
        while i < len(sent):
            slice_ = sent[i : i + approx_chars]
            pieces.append(
                (slice_, start + local_start + i, start + local_start + i + len(slice_))
            )
            i += approx_chars
    return pieces


def _blocks_to_units(blocks: list[Block]) -> list[_Unit]:
    units: list[_Unit] = []
    offset = 0
    for i, block in enumerate(blocks):
        if i > 0:
            offset += 2  # matches the "\n\n" join between block texts
        for para, start, end in _split_paragraphs(block.text):
            units.append(
                _Unit(
                    text=para,
                    page=block.page,
                    page_hash=block.page_hash,
                    section_path=block.section_path,
                    start=offset + start,
                    end=offset + end,
                    tokens=count_tokens(para),
                )
            )
        offset += len(block.text)
    return units


def _expand_oversized(units: list[_Unit], chunk_size_tokens: int) -> list[_Unit]:
    out: list[_Unit] = []
    for u in units:
        if u.tokens <= chunk_size_tokens:
            out.append(u)
            continue
        for text, start, end in _split_oversized(u.text, u.start, chunk_size_tokens):
            out.append(
                _Unit(
                    text=text,
                    page=u.page,
                    page_hash=u.page_hash,
                    section_path=u.section_path,
                    start=start,
                    end=end,
                    tokens=count_tokens(text),
                )
            )
    return out


def _derive_chapter_section(
    section_path: tuple[str, ...]
) -> tuple[str | None, str | None]:
    chapter_title = section_path[0] if section_path else None
    section = section_path[-1] if len(section_path) > 1 else None
    return chapter_title, section


class StructuralChunker:
    """Default Chunker: groups by chapter, chunks by token budget with
    overlap, never merges content across a chapter boundary."""

    def __init__(
        self,
        *,
        chunk_size_tokens: int = 400,
        chunk_overlap_tokens: int = 50,
        max_group_tokens: int = 30_000,
    ) -> None:
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.max_group_tokens = max_group_tokens

    def chunk(self, doc: Document) -> list[ChunkGroup]:
        chapters = self._split_into_chapters(doc.blocks)

        groups: list[ChunkGroup] = []
        chunk_index = 0
        group_ordinal = 0

        for chapter_index, chapter_blocks in enumerate(chapters):
            for sub_blocks in self._split_by_group_budget(chapter_blocks):
                units = _expand_oversized(
                    _blocks_to_units(sub_blocks), self.chunk_size_tokens
                )
                if not units:
                    continue
                group_id = f"{doc.source_id}:group:{group_ordinal}"
                group_ordinal += 1

                chunks: list[Chunk] = []
                for unit_batch in self._pack_units(units):
                    text = "\n\n".join(u.text for u in unit_batch)
                    pages = [u.page for u in unit_batch if u.page is not None]
                    section_path = unit_batch[0].section_path
                    chapter_title, section = _derive_chapter_section(section_path)

                    # A chunk gets a page-level dedupe hash only when every
                    # unit in it comes from the same single page's Block —
                    # a chunk straddling two pages has no one page whose
                    # content it fully represents, so hashing it would
                    # claim a false-precision dedupe signal.
                    page_hashes = {u.page_hash for u in unit_batch if u.page_hash is not None}
                    extra = {"page_hash": next(iter(page_hashes))} if len(page_hashes) == 1 else {}

                    meta = ChunkMeta(
                        source_id=doc.source_id,
                        source_path=doc.path,
                        source_title=doc.title,
                        source_type=doc.source_type,
                        section_path=section_path,
                        chapter_title=chapter_title,
                        chapter_index=chapter_index if section_path else None,
                        section=section,
                        page_start=min(pages) if pages else None,
                        page_end=max(pages) if pages else None,
                        chunk_index=chunk_index,
                        context_group_id=group_id,
                        char_start=unit_batch[0].start,
                        char_end=unit_batch[-1].end,
                        token_count=count_tokens(text),
                        extra=extra,
                    )
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc.source_id}:{chunk_index}",
                            text=text,
                            meta=meta,
                        )
                    )
                    chunk_index += 1

                groups.append(ChunkGroup(group_id=group_id, chunks=chunks))

        return groups

    def _split_into_chapters(self, blocks: list[Block]) -> list[list[Block]]:
        """Group blocks into runs sharing the same top-level section_path
        element, preserving document order. Blocks with no section_path
        (e.g. a whole .txt file, or front matter before the first heading)
        form their own chapter-less run."""
        chapters: list[list[Block]] = []
        current_key: object = object()  # sentinel, never equals a real key
        for block in blocks:
            key = block.section_path[0] if block.section_path else None
            if not chapters or key != current_key:
                chapters.append([])
                current_key = key
            chapters[-1].append(block)
        return chapters

    def _split_by_group_budget(self, blocks: list[Block]) -> list[list[Block]]:
        """Split a chapter's blocks into sub-groups, each under
        max_group_tokens total, without splitting a block across groups."""
        groups: list[list[Block]] = []
        current: list[Block] = []
        current_tokens = 0
        for block in blocks:
            block_tokens = count_tokens(block.text)
            if current and current_tokens + block_tokens > self.max_group_tokens:
                groups.append(current)
                current, current_tokens = [], 0
            current.append(block)
            current_tokens += block_tokens
        if current:
            groups.append(current)
        return groups or [[]]

    def _pack_units(self, units: list[_Unit]) -> list[list[_Unit]]:
        """Greedily pack units into chunks up to chunk_size_tokens, then
        start the next chunk with however many trailing units cover
        chunk_overlap_tokens.

        A change in `section_path` is *also* a hard chunk boundary, on top
        of the token budget: a chunk's `ChunkMeta.section_path` is taken
        from its first unit (see `chunk()`), so silently packing a
        subsection's opening paragraphs in with the end of the previous
        subsection would mislabel — or lose — that subsection's tag. Since
        a boundary crossing never carries overlap (the trailing content
        belongs to a different section), overlap is only computed when the
        flush was purely for token budget.

        Each iteration either appends `u` to the batch or flushes — never
        both without also advancing `i` — so a unit is placed in exactly
        one step. If the overlap carried into a fresh batch would itself
        overflow against the next unit, the overlap is dropped rather than
        re-flushed, which guarantees termination and avoids emitting a
        spurious overlap-only micro-chunk.
        """
        batches: list[list[_Unit]] = []
        batch: list[_Unit] = []
        batch_tokens = 0

        for u in units:
            section_changed = bool(batch) and batch[-1].section_path != u.section_path
            over_budget = bool(batch) and batch_tokens + u.tokens > self.chunk_size_tokens
            if section_changed or over_budget:
                batches.append(batch)
                overlap: list[_Unit] = []
                overlap_tokens = 0
                if not section_changed:
                    for ou in reversed(batch):
                        if overlap_tokens + ou.tokens > self.chunk_overlap_tokens:
                            break
                        overlap.insert(0, ou)
                        overlap_tokens += ou.tokens
                    if overlap_tokens + u.tokens > self.chunk_size_tokens:
                        overlap, overlap_tokens = [], 0
                batch, batch_tokens = list(overlap), overlap_tokens
            batch.append(u)
            batch_tokens += u.tokens

        if batch:
            batches.append(batch)
        return batches
