"""SemanticChunker — embedding-boundary chunking (the second Chunker
implementation, see ragcore/chunking/semantic.py). Builds Document/Block
directly rather than going through a Loader, for exact control over
sentence content and section_path.
"""

from __future__ import annotations

from typing import Callable, Literal, Sequence

import pytest

from ragcore.chunking.semantic import SemanticChunker
from ragcore.types import Block, Document, Vector


class KeyedEmbedder:
    """Deterministic Embedder double: `vec_fn(sentence_text)` decides each
    sentence's vector, so a test can engineer an exact similarity profile
    instead of relying on real semantic embeddings."""

    name = "keyed-test-embedder"
    dim = 2

    def __init__(self, vec_fn: Callable[[str], Vector]) -> None:
        self._vec_fn = vec_fn
        self.embed_calls: list[list[str]] = []

    def embed_groups(
        self,
        groups: Sequence[Sequence[str]],
        *,
        input_type: Literal["document", "query", None] = "document",
    ) -> list[list[Vector]]:
        out = []
        for group in groups:
            self.embed_calls.append(list(group))
            out.append([self._vec_fn(t) for t in group])
        return out

    def embed_query(self, text: str) -> Vector:
        return self._vec_fn(text)


def _doc(blocks: list[Block], source_id: str = "docsrc") -> Document:
    return Document(
        source_id=source_id, path="book.txt", title="Book", source_type="txt", blocks=blocks
    )


def test_breaks_at_a_semantic_shift() -> None:
    # Two "apple" sentences (near-identical vectors) followed by two
    # "rocket" sentences (orthogonal vector) — a clean, unambiguous
    # similarity drop between sentence 2 and sentence 3.
    text = (
        "Apples are a popular fruit. Apples grow well in temperate climates. "
        "Rockets use controlled combustion. Rockets can reach orbital velocity."
    )
    doc = _doc([Block(text=text, page=None, section_path=("Chapter 1",))])

    def vec_fn(sentence: str) -> Vector:
        return [1.0, 0.0] if "Apple" in sentence else [0.0, 1.0]

    embedder = KeyedEmbedder(vec_fn)
    chunker = SemanticChunker(embedder, breakpoint_threshold=0.5)

    [group] = chunker.chunk(doc)
    assert len(group.chunks) == 2
    assert "Apples" in group.chunks[0].text and "Rockets" not in group.chunks[0].text
    assert "Rockets" in group.chunks[1].text and "Apples" not in group.chunks[1].text


def test_high_similarity_throughout_yields_one_chunk_under_budget() -> None:
    text = "Apples are a fruit. Apples are grown worldwide. Apples come in many colors."
    doc = _doc([Block(text=text, page=None, section_path=("Chapter 1",))])

    embedder = KeyedEmbedder(lambda s: [1.0, 0.0])  # every sentence identical
    chunker = SemanticChunker(embedder, breakpoint_threshold=0.5, chunk_size_tokens=400)

    [group] = chunker.chunk(doc)
    assert len(group.chunks) == 1


def test_token_budget_forces_a_split_despite_high_similarity() -> None:
    sentence = "Apples are a wonderful and extremely popular fruit grown worldwide. "
    text = sentence * 40  # identical vectors throughout, but a lot of tokens
    doc = _doc([Block(text=text, page=None, section_path=("Chapter 1",))])

    embedder = KeyedEmbedder(lambda s: [1.0, 0.0])
    chunker = SemanticChunker(
        embedder, breakpoint_threshold=0.0, chunk_size_tokens=40, chunk_overlap_tokens=5
    )

    [group] = chunker.chunk(doc)
    assert len(group.chunks) > 1
    for c in group.chunks:
        assert c.meta.token_count <= 40 + 5  # budget plus a bounded overlap carry-in


def test_overlap_not_carried_across_a_semantic_breakpoint() -> None:
    text = "Apples are a fruit. Apples taste sweet. Rockets fly to orbit. Rockets burn fuel."
    doc = _doc([Block(text=text, page=None, section_path=("Chapter 1",))])

    def vec_fn(sentence: str) -> Vector:
        return [1.0, 0.0] if "Apple" in sentence else [0.0, 1.0]

    embedder = KeyedEmbedder(vec_fn)
    chunker = SemanticChunker(embedder, breakpoint_threshold=0.5)

    [group] = chunker.chunk(doc)
    assert len(group.chunks) == 2
    assert "Apples" not in group.chunks[1].text  # no apple-sentence overlap leaked in


def test_section_change_forces_a_new_chunk_even_with_identical_vectors() -> None:
    blocks = [
        Block(text="Apples are a fruit.", page=None, section_path=("Chapter 1",)),
        Block(text="Apples taste sweet.", page=None, section_path=("Chapter 1", "Sub")),
    ]
    doc = _doc(blocks)

    embedder = KeyedEmbedder(lambda s: [1.0, 0.0])  # identical throughout
    chunker = SemanticChunker(embedder, breakpoint_threshold=0.0)

    [group] = chunker.chunk(doc)
    assert len(group.chunks) == 2
    assert group.chunks[0].meta.section_path == ("Chapter 1",)
    assert group.chunks[1].meta.section_path == ("Chapter 1", "Sub")


def test_oversized_single_sentence_hard_splits() -> None:
    long_sentence = "word " * 2000 + "."
    doc = _doc([Block(text=long_sentence, page=None, section_path=("Chapter 1",))])

    embedder = KeyedEmbedder(lambda s: [1.0, 0.0])
    chunker = SemanticChunker(embedder, chunk_size_tokens=50, chunk_overlap_tokens=5)

    [group] = chunker.chunk(doc)
    assert len(group.chunks) > 1
    for c in group.chunks:
        assert c.meta.token_count <= 50


def test_min_sentences_per_chunk_suppresses_early_breakpoints() -> None:
    text = "Apples are great. Rockets are fast. Rockets are loud. Rockets are expensive."
    doc = _doc([Block(text=text, page=None, section_path=("Chapter 1",))])

    def vec_fn(sentence: str) -> Vector:
        return [1.0, 0.0] if "Apple" in sentence else [0.0, 1.0]

    embedder = KeyedEmbedder(vec_fn)
    chunker = SemanticChunker(embedder, breakpoint_threshold=0.5, min_sentences_per_chunk=3)

    [group] = chunker.chunk(doc)
    # The breakpoint after sentence 1 is suppressed (batch has < 3
    # sentences yet); the first chunk must absorb at least 3 sentences.
    assert group.chunks[0].meta.token_count > 0
    assert "Rockets are loud" in group.chunks[0].text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size_tokens": 10, "chunk_overlap_tokens": 10},
        {"breakpoint_threshold": 1.5},
        {"breakpoint_threshold": -0.1},
    ],
)
def test_invalid_config_raises(kwargs: dict) -> None:
    embedder = KeyedEmbedder(lambda s: [1.0, 0.0])
    with pytest.raises(ValueError):
        SemanticChunker(embedder, **kwargs)
