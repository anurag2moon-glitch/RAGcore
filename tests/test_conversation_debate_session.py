"""DebateSession: shared transcript across personas, seed_topic(), unknown
persona/duplicate-Corpus guards, and — the critical test — that a persona's
generation step never receives chunks belonging to the opponent's corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from ragcore.conversation.session import DebateSession
from ragcore.conversation.transcript import InMemoryTranscriptStore
from ragcore.conversation.types import Turn
from ragcore.corpus import Corpus
from ragcore.errors import SessionError
from tests.fakes import FakeEmbedder, FakeGenerator, InMemoryStore


class _FakeRewriter:
    model = "fake-rewriter"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Turn, ...], str]] = []

    def rewrite(self, *, transcript: Sequence[Turn], speaking_as: str) -> str:
        self.calls.append((tuple(transcript), speaking_as))
        return transcript[-1].text


def _persona_corpus(tmp_path: Path, name: str, marker_word: str) -> Corpus:
    book = tmp_path / f"{name}.md"
    book.write_text(f"# On the topic\n\nMy own view relies on {marker_word} above all.\n")
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(book)
    corpus.generator = FakeGenerator()
    return corpus


def test_seed_topic_appears_first_in_the_shared_transcript(tmp_path: Path) -> None:
    a = _persona_corpus(tmp_path, "a", "reasonword")
    b = _persona_corpus(tmp_path, "b", "willword")
    debate = DebateSession(
        {"a": a, "b": b}, transcript_store=InMemoryTranscriptStore(), rewriter=_FakeRewriter()
    )

    debate.seed_topic("Is virtue teachable?")

    assert [t.speaker for t in debate.transcript()] == ["topic"]
    assert debate.transcript()[0].text == "Is virtue teachable?"


def test_take_turn_appends_the_speakers_reply_to_the_shared_transcript(tmp_path: Path) -> None:
    a = _persona_corpus(tmp_path, "a", "reasonword")
    b = _persona_corpus(tmp_path, "b", "willword")
    debate = DebateSession(
        {"a": a, "b": b}, transcript_store=InMemoryTranscriptStore(), rewriter=_FakeRewriter()
    )
    debate.seed_topic("Is virtue teachable?")

    debate.take_turn("a")
    debate.take_turn("b")

    assert [t.speaker for t in debate.transcript()] == ["topic", "a", "b"]


def test_take_turn_with_unknown_persona_raises_session_error(tmp_path: Path) -> None:
    a = _persona_corpus(tmp_path, "a", "reasonword")
    b = _persona_corpus(tmp_path, "b", "willword")
    debate = DebateSession(
        {"a": a, "b": b}, transcript_store=InMemoryTranscriptStore(), rewriter=_FakeRewriter()
    )
    debate.seed_topic("topic")

    with pytest.raises(SessionError, match="unknown persona_id"):
        debate.take_turn("c")


def test_two_personas_sharing_the_same_corpus_instance_raises_at_construction(tmp_path: Path) -> None:
    a = _persona_corpus(tmp_path, "a", "reasonword")

    with pytest.raises(SessionError, match="share the same Corpus instance"):
        DebateSession(
            {"a": a, "b": a}, transcript_store=InMemoryTranscriptStore(), rewriter=_FakeRewriter()
        )


def test_close_clears_the_shared_session(tmp_path: Path) -> None:
    a = _persona_corpus(tmp_path, "a", "reasonword")
    b = _persona_corpus(tmp_path, "b", "willword")
    store = InMemoryTranscriptStore()
    debate = DebateSession({"a": a, "b": b}, transcript_store=store, rewriter=_FakeRewriter())
    debate.seed_topic("topic")
    session_id = debate.session_id

    debate.close()

    assert store.recent_turns(session_id) == []


def test_a_persona_never_receives_chunks_from_the_opponents_corpus(tmp_path: Path) -> None:
    """The critical isolation test: each persona's generator must only ever
    be handed chunk_ids that actually exist in that persona's own store —
    never the opponent's, despite the opponent's prior turn being visible
    as conversational context in the shared transcript."""
    a = _persona_corpus(tmp_path, "a", "reasonword")
    b = _persona_corpus(tmp_path, "b", "willword")
    a_chunk_ids = {c.chunk_id for c in a.store.all_chunks()}
    b_chunk_ids = {c.chunk_id for c in b.store.all_chunks()}
    assert a_chunk_ids and b_chunk_ids and a_chunk_ids.isdisjoint(b_chunk_ids)

    rewriter = _FakeRewriter()
    debate = DebateSession(
        {"a": a, "b": b}, transcript_store=InMemoryTranscriptStore(), rewriter=rewriter
    )
    debate.seed_topic("Is virtue teachable?")

    for i in range(6):
        debate.take_turn("a" if i % 2 == 0 else "b")

    for _query, chunk_ids in a.generator.calls:
        seen = set(chunk_ids)
        assert seen <= a_chunk_ids, f"persona 'a' was handed chunks outside its own corpus: {seen - a_chunk_ids}"
        assert seen.isdisjoint(b_chunk_ids), "persona 'a' was handed a chunk from persona 'b's corpus"

    for _query, chunk_ids in b.generator.calls:
        seen = set(chunk_ids)
        assert seen <= b_chunk_ids, f"persona 'b' was handed chunks outside its own corpus: {seen - b_chunk_ids}"
        assert seen.isdisjoint(a_chunk_ids), "persona 'b' was handed a chunk from persona 'a's corpus"

    # And the opponent's previous turn genuinely was visible as
    # conversational context to the rewriter (never treated as retrievable
    # source material — it never appears in either corpus's own store).
    _transcript_seen_by_second_call, _speaking_as = rewriter.calls[1]
    assert any(t.speaker == "a" for t in _transcript_seen_by_second_call)
