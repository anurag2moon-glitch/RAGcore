"""SemanticChunker — an embedding-boundary Chunker, the second strategy
behind the Chunker protocol (see ragcore/chunking/base.py). Proves
"pluggable and configurable per document type" (the long-range build
spec's Phase 1 requirement) is real, not just protocol-shaped with one
implementation.

Where StructuralChunker packs paragraphs by token budget alone,
SemanticChunker splits each chapter into sentences, embeds every sentence
(one `Embedder.embed_groups()` call per chapter, same contextualization
granularity StructuralChunker's groups already use), and starts a new
chunk wherever consecutive sentences' cosine similarity drops below
`breakpoint_threshold` — a real semantic-shift signal, not a token-count
heuristic. The token budget (`chunk_size_tokens`) is still a hard ceiling
regardless of similarity, so a long run of highly-similar sentences can't
produce an unbounded chunk.

**Real cost tradeoff, stated plainly**: this embeds every sentence at
chunk time, then Corpus.ingest() embeds the resulting chunks again right
after — semantic chunking is strictly more expensive per document than
StructuralChunker. This is the known, accepted cost of embedding-boundary
chunking (the same tradeoff LlamaIndex's SemanticSplitterNodeParser and
similar techniques make); it is not a bug to "fix" by caching the sentence
vectors for reuse as chunk vectors, since a chunk's vector represents the
*whole chunk's* text, not one of its sentences.

Chapter-splitting and group-budget-splitting duplicate
StructuralChunker's private `_split_into_chapters`/`_split_by_group_budget`
logic (same small algorithm, deliberately not extracted into a shared
module) rather than refactoring StructuralChunker to share it — this
keeps StructuralChunker, the default and most heavily relied-on
implementation, untouched by this addition. `_split_oversized` (the
"even one sentence is too large" hard-character-slice fallback) *is*
imported from `ragcore.chunking.structural` rather than duplicated: it's
a pure, already-tested function with no coupling to StructuralChunker's
state, so importing it carries none of that risk.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ragcore.chunking.structural import _split_oversized
from ragcore.chunking.tokenizer import count_tokens
from ragcore.embedding.base import Embedder
from ragcore.types import Block, Chunk, ChunkGroup, ChunkMeta, Document, Vector

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _derive_chapter_section(section_path: tuple[str, ...]) -> tuple[str | None, str | None]:
    chapter_title = section_path[0] if section_path else None
    section = section_path[-1] if len(section_path) > 1 else None
    return chapter_title, section


def _split_into_chapters(blocks: list[Block]) -> list[list[Block]]:
    chapters: list[list[Block]] = []
    current_key: object = object()
    for block in blocks:
        key = block.section_path[0] if block.section_path else None
        if not chapters or key != current_key:
            chapters.append([])
            current_key = key
        chapters[-1].append(block)
    return chapters


def _split_by_group_budget(blocks: list[Block], max_group_tokens: int) -> list[list[Block]]:
    groups: list[list[Block]] = []
    current: list[Block] = []
    current_tokens = 0
    for block in blocks:
        block_tokens = count_tokens(block.text)
        if current and current_tokens + block_tokens > max_group_tokens:
            groups.append(current)
            current, current_tokens = [], 0
        current.append(block)
        current_tokens += block_tokens
    if current:
        groups.append(current)
    return groups or [[]]


def _cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass(frozen=True)
class _Sentence:
    text: str
    page: int | None
    page_hash: str | None
    section_path: tuple[str, ...]
    start: int
    end: int
    tokens: int


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    """(stripped_sentence, start, end) triples, start/end exact offsets
    into `text`. Whitespace-only gaps between sentences are dropped, same
    convention as structural.py's `_split_paragraphs`."""
    results: list[tuple[str, int, int]] = []
    pos = 0
    for raw in _SENTENCE_BOUNDARY.split(text):
        if not raw:
            continue
        local_start = text.index(raw, pos)
        pos = local_start + len(raw)
        stripped = raw.strip()
        if not stripped:
            continue
        lstrip_len = len(raw) - len(raw.lstrip())
        real_start = local_start + lstrip_len
        results.append((stripped, real_start, real_start + len(stripped)))
    return results


def _blocks_to_sentences(blocks: list[Block]) -> list[_Sentence]:
    sentences: list[_Sentence] = []
    offset = 0
    for i, block in enumerate(blocks):
        if i > 0:
            offset += 2  # matches the "\n\n" join between block texts
        for text, start, end in _split_sentences(block.text):
            sentences.append(
                _Sentence(
                    text=text,
                    page=block.page,
                    page_hash=block.page_hash,
                    section_path=block.section_path,
                    start=offset + start,
                    end=offset + end,
                    tokens=count_tokens(text),
                )
            )
        offset += len(block.text)
    return sentences


def _expand_oversized(sentences: list[_Sentence], chunk_size_tokens: int) -> list[_Sentence]:
    out: list[_Sentence] = []
    for s in sentences:
        if s.tokens <= chunk_size_tokens:
            out.append(s)
            continue
        for text, start, end in _split_oversized(s.text, s.start, chunk_size_tokens):
            out.append(
                _Sentence(
                    text=text,
                    page=s.page,
                    page_hash=s.page_hash,
                    section_path=s.section_path,
                    start=start,
                    end=end,
                    tokens=count_tokens(text),
                )
            )
    return out


class SemanticChunker:
    """Chunker that starts a new chunk at a semantic-similarity drop
    between consecutive sentences, rather than only at a token-budget
    boundary. See module docstring for the full rationale and cost
    tradeoff.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        breakpoint_threshold: float = 0.75,
        min_sentences_per_chunk: int = 1,
        chunk_size_tokens: int = 400,
        chunk_overlap_tokens: int = 50,
        max_group_tokens: int = 30_000,
    ) -> None:
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        if not 0.0 <= breakpoint_threshold <= 1.0:
            raise ValueError("breakpoint_threshold must be within [0.0, 1.0]")
        self.embedder = embedder
        self.breakpoint_threshold = breakpoint_threshold
        self.min_sentences_per_chunk = max(1, min_sentences_per_chunk)
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.max_group_tokens = max_group_tokens

    def chunk(self, doc: Document) -> list[ChunkGroup]:
        chapters = _split_into_chapters(doc.blocks)

        groups: list[ChunkGroup] = []
        chunk_index = 0
        group_ordinal = 0

        for chapter_index, chapter_blocks in enumerate(chapters):
            for sub_blocks in _split_by_group_budget(chapter_blocks, self.max_group_tokens):
                sentences = _expand_oversized(
                    _blocks_to_sentences(sub_blocks), self.chunk_size_tokens
                )
                if not sentences:
                    continue

                group_id = f"{doc.source_id}:group:{group_ordinal}"
                group_ordinal += 1

                vectors = self.embedder.embed_groups(
                    [[s.text for s in sentences]], input_type="document"
                )[0]

                chunks: list[Chunk] = []
                for batch in self._pack_by_breakpoints(sentences, vectors):
                    text = "\n\n".join(s.text for s in batch)
                    pages = [s.page for s in batch if s.page is not None]
                    section_path = batch[0].section_path
                    chapter_title, section = _derive_chapter_section(section_path)

                    page_hashes = {s.page_hash for s in batch if s.page_hash is not None}
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
                        char_start=batch[0].start,
                        char_end=batch[-1].end,
                        token_count=count_tokens(text),
                        extra=extra,
                    )
                    chunks.append(
                        Chunk(chunk_id=f"{doc.source_id}:{chunk_index}", text=text, meta=meta)
                    )
                    chunk_index += 1

                groups.append(ChunkGroup(group_id=group_id, chunks=chunks))

        return groups

    def _pack_by_breakpoints(
        self, sentences: list[_Sentence], vectors: list[Vector]
    ) -> list[list[_Sentence]]:
        """Greedily accumulate sentences into a chunk until whichever comes
        first: a section change, the token budget, or a semantic
        breakpoint (cosine similarity to the previous sentence falling
        below `breakpoint_threshold`) once the chunk already has at least
        `min_sentences_per_chunk` sentences. Overlap is only carried across
        a token-budget flush — a section change or a genuine semantic
        breakpoint both mean the next chunk's content is meaningfully
        different, so carrying trailing context across either would mix
        in material the breakpoint signal says doesn't belong together
        (the same reasoning ragcore.chunking.structural.StructuralChunker
        already applies to section changes).
        """
        batches: list[list[_Sentence]] = []
        batch: list[_Sentence] = []
        batch_tokens = 0

        for i, s in enumerate(sentences):
            if not batch:
                batch.append(s)
                batch_tokens = s.tokens
                continue

            section_changed = batch[-1].section_path != s.section_path
            over_budget = batch_tokens + s.tokens > self.chunk_size_tokens
            similarity = _cosine(vectors[i - 1], vectors[i])
            is_breakpoint = (
                not section_changed
                and not over_budget
                and similarity < self.breakpoint_threshold
                and len(batch) >= self.min_sentences_per_chunk
            )

            if section_changed or over_budget or is_breakpoint:
                batches.append(batch)
                overlap: list[_Sentence] = []
                overlap_tokens = 0
                if over_budget and not section_changed:
                    for prev in reversed(batch):
                        if overlap_tokens + prev.tokens > self.chunk_overlap_tokens:
                            break
                        overlap.insert(0, prev)
                        overlap_tokens += prev.tokens
                    if overlap_tokens + s.tokens > self.chunk_size_tokens:
                        overlap, overlap_tokens = [], 0
                batch, batch_tokens = list(overlap), overlap_tokens

            batch.append(s)
            batch_tokens += s.tokens

        if batch:
            batches.append(batch)
        return batches
