"""app.ci_fakes: the offline, deterministic Embedder+Generator pair
the CI integration test (docs/adr/0022) uses to exercise a real
running Compose stack without a VOYAGE_API_KEY/ANTHROPIC_API_KEY
secret. Verified for real, not just by inspection: two texts sharing
no vocabulary score zero cosine similarity, two sharing vocabulary
score meaningfully above zero - a constant embedding (the simpler
tests/fakes.FakeEmbedder, fine for in-process tests that never check
*which* chunk was retrieved) would make retrieval order arbitrary,
which this CI test must not accept.
"""

from __future__ import annotations

from app.ci_fakes import EchoCitingGenerator, HashingEmbedder

from ragcore.types import Chunk as RagcoreChunk
from ragcore.types import ChunkMeta, Retrieved


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hashing_embedder_scores_shared_vocabulary_above_zero() -> None:
    embedder = HashingEmbedder()
    query = embedder.embed_query("Tell me about foxes")
    fox_text = embedder.embed_query("The quick brown fox jumps over the lazy dog")
    assert _cosine(query, fox_text) > 0.0


def test_hashing_embedder_scores_unrelated_text_at_zero() -> None:
    embedder = HashingEmbedder()
    query = embedder.embed_query("Tell me about foxes")
    unrelated = embedder.embed_query("Quantum computers use qubits for computation")
    assert _cosine(query, unrelated) == 0.0


def test_hashing_embedder_is_deterministic() -> None:
    embedder = HashingEmbedder()
    assert embedder.embed_query("same text every time") == embedder.embed_query("same text every time")


def test_hashing_embedder_satisfies_the_embedder_protocol() -> None:
    from ragcore.embedding.base import Embedder

    embedder = HashingEmbedder()
    assert isinstance(embedder, Embedder)


def test_hashing_embedder_embed_groups_preserves_shape() -> None:
    embedder = HashingEmbedder()
    result = embedder.embed_groups([["a", "b"], ["c"]])
    assert [len(group) for group in result] == [2, 1]
    assert all(len(vec) == embedder.dim for group in result for vec in group)


def test_echo_citing_generator_cites_every_context_chunk() -> None:
    generator = EchoCitingGenerator()
    meta = ChunkMeta(
        source_id="s1", source_path="/tmp/a.txt", source_title=None, source_type="txt",
        chapter_index=0, chapter_title=None, section=None, section_path=(), page_start=None,
        page_end=None, chunk_index=0, context_group_id="s1:0", char_start=0, char_end=9, token_count=2,
    )
    chunk = RagcoreChunk(chunk_id="c1", text="some text", meta=meta)
    context = [Retrieved(chunk=chunk, score=0.9)]

    answer = generator.generate("a query", context)

    assert "a query" in answer.text
    assert answer.citations == context
