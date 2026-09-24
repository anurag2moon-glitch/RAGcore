"""Live smoke tests against the real Claude API for the concept layer.
Excluded by default (see pyproject.toml `addopts = "-m 'not live'"`); run
explicitly with `pytest -m live`. Requires ANTHROPIC_API_KEY.

Uses the same real Chapter III passage as test_live_claude_extraction.py
(the Bingley/Darcy ballroom contrast), which discusses a real abstract
theme — social ease versus pride, and how first impressions of character
form — so a real extraction has genuine abstract propositions to find, not
scaffolding.
"""

from __future__ import annotations

import os

import pytest

from ragcore.types import Chunk, ChunkMeta

pytestmark = pytest.mark.live


def _chapter_iii_chunks() -> list[Chunk]:
    def meta(idx: int, text_len: int) -> ChunkMeta:
        return ChunkMeta(
            source_id="src1",
            source_path="pride-and-prejudice.epub",
            source_title="Pride and Prejudice",
            source_type="epub",
            section_path=("He rode a black horse. CHAPTER III.",),
            chapter_title="He rode a black horse. CHAPTER III.",
            chapter_index=4,
            section=None,
            page_start=None,
            page_end=None,
            chunk_index=idx,
            context_group_id="src1:group:4",
            char_start=0,
            char_end=text_len,
            token_count=text_len // 4,
        )

    text_35 = (
        "Mr. Bingley had soon made himself acquainted with all the principal "
        "people in the room: he was lively and unreserved, danced every dance, "
        "was angry that the ball closed so early, and talked of giving one "
        "himself at Netherfield. Such amiable qualities must speak for "
        "themselves. What a contrast between him and his friend! Mr. Darcy "
        "danced only once with Mrs. Hurst and once with Miss Bingley, declined "
        "being introduced to any other lady, and spent the rest of the evening "
        "in walking about the room, speaking occasionally to one of his own "
        "party. His character was decided. He was the proudest, most "
        "disagreeable man in the world, and everybody hoped that he would "
        "never come there again."
    )
    text_36 = (
        '"You are dancing with the only handsome girl in the room," said Mr. '
        'Darcy, looking at the eldest Miss Bennet. "She is tolerable: but not '
        "handsome enough to tempt me; and I am in no humour at present to give "
        "consequence to young ladies who are slighted by other men. You had "
        'better return to your partner and enjoy her smiles, for you are '
        'wasting your time with me." Mr. Bingley followed his advice. Mr. '
        "Darcy walked off; and Elizabeth remained with no very cordial "
        "feelings towards him."
    )
    return [
        Chunk(chunk_id="src1:35", text=text_35, meta=meta(35, len(text_35))),
        Chunk(chunk_id="src1:36", text=text_36, meta=meta(36, len(text_36))),
    ]


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_claude_concept_extractor_finds_abstract_propositions_not_keywords() -> None:
    from ragcore.concepts.claude_extractor import ClaudeConceptExtractor

    extractor = ClaudeConceptExtractor()
    concepts, edges = extractor.extract(_chapter_iii_chunks())

    assert len(concepts) >= 2
    # A real extraction should never regress to bare keywords like "pride"
    # or "Darcy" — every label should be a multi-word proposition.
    for c in concepts:
        assert len(c.label.split()) >= 3, f"concept label reads like a keyword: {c.label!r}"

    valid_chunk_ids = {"src1:35", "src1:36"}
    for c in concepts:
        assert set(c.source_chunk_ids) <= valid_chunk_ids
    for e in edges:
        assert set(e.source_chunk_ids) <= valid_chunk_ids


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_claude_concept_mapper_maps_a_relevant_query() -> None:
    from ragcore.concepts.mapper import ClaudeConceptMapper
    from ragcore.concepts.types import Concept

    concepts = [
        Concept(
            concept_id="concept:pride-and-first-impressions",
            label="pride distorting first impressions of character",
            description="Darcy's aloofness at the ball is read as pride rather than shyness.",
            source_chunk_ids=("src1:35", "src1:36"),
        ),
        Concept(
            concept_id="concept:unrelated",
            label="the economics of 19th century wheat tariffs",
            description="Completely unrelated to the ballroom scene.",
            source_chunk_ids=(),
        ),
    ]

    class _AllShortlisted:
        def map(self, query, concepts):
            return [c.concept_id for c in concepts]

    mapper = ClaudeConceptMapper(_AllShortlisted())
    mapped = mapper.map("why did people think Mr. Darcy was proud", concepts)

    assert "concept:pride-and-first-impressions" in mapped
    assert "concept:unrelated" not in mapped


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_claude_verifier_flags_an_unsupported_injected_claim() -> None:
    from ragcore.generation.base import Answer
    from ragcore.types import Retrieved
    from ragcore.verification.claude import ClaudeVerifier

    chunk = _chapter_iii_chunks()[0]
    answer = Answer(
        text=(
            "Mr. Bingley was lively and danced every dance at the ball. "
            "Mr. Darcy secretly ran a smuggling operation out of Netherfield."
        ),
        citations=[Retrieved(chunk=chunk, score=0.9)],
    )

    verifier = ClaudeVerifier()
    verification = verifier.verify("what happened at the ball", answer)

    assert len(verification.unsupported) >= 1
    assert any("smuggling" in c.claim.lower() for c in verification.unsupported)
