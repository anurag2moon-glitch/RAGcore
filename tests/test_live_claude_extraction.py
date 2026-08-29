"""Live smoke test against the real Claude API. Excluded by default (see
pyproject.toml `addopts = "-m 'not live'"`); run explicitly with
`pytest -m live`. Requires ANTHROPIC_API_KEY in the environment.

Uses the real Chapter III text (the Bingley/Darcy contrast passage and the
"she is tolerable" exchange) that motivated the GraphRAG design doc — see
the design doc's GraphRAG addendum for the real chunk_ids this was pulled
from in the actual ingested corpus.
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
def test_claude_entity_extractor_recovers_darcy_bingley_relation() -> None:
    from ragcore.graph.claude_extractor import ClaudeEntityExtractor

    extractor = ClaudeEntityExtractor()
    entities, relations = extractor.extract(_chapter_iii_chunks())

    names = {e.name for e in entities}
    assert any("Darcy" in n for n in names)
    assert any("Bingley" in n for n in names)

    # every entity/relation must cite one of the two real chunk_ids given
    valid_chunk_ids = {"src1:35", "src1:36"}
    for e in entities:
        assert set(e.source_chunk_ids) <= valid_chunk_ids
    for r in relations:
        assert set(r.source_chunk_ids) <= valid_chunk_ids

    relation_types = {(r.relation_type, r.source_entity_id, r.target_entity_id) for r in relations}
    darcy_id = next(e.entity_id for e in entities if "Darcy" in e.name)
    bingley_id = next(e.entity_id for e in entities if "Bingley" in e.name)
    assert any(
        {darcy_id, bingley_id} == {src, tgt} for (_type, src, tgt) in relation_types
    ), f"expected a Darcy<->Bingley relation, got: {relation_types}"
