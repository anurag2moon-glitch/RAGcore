"""PersonaChatSession: the user's message is appended before rewriting, the
REWRITTEN query (not the raw message) reaches the persona's Corpus/Generator,
the persona's reply is appended as plain text, and close() clears the
session from the transcript store.
"""

from __future__ import annotations

from typing import Sequence

from ragcore.conversation.session import PersonaChatSession
from ragcore.conversation.transcript import InMemoryTranscriptStore
from ragcore.conversation.types import Turn
from ragcore.corpus import Corpus
from tests.fakes import FakeEmbedder, FakeGenerator, InMemoryStore


class _FakeRewriter:
    model = "fake-rewriter"

    def __init__(self, *, prefix: str = "REWRITTEN: ") -> None:
        self._prefix = prefix
        self.calls: list[tuple[tuple[Turn, ...], str]] = []

    def rewrite(self, *, transcript: Sequence[Turn], speaking_as: str) -> str:
        self.calls.append((tuple(transcript), speaking_as))
        return self._prefix + transcript[-1].text


def _corpus() -> Corpus:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.generator = FakeGenerator()
    return corpus


def test_send_appends_the_user_message_before_rewriting() -> None:
    corpus = _corpus()
    rewriter = _FakeRewriter()
    session = PersonaChatSession(
        "socrates", corpus, transcript_store=InMemoryTranscriptStore(), rewriter=rewriter
    )

    session.send("what is virtue?")

    seen_transcript, speaking_as = rewriter.calls[0]
    assert [t.speaker for t in seen_transcript] == ["user"]
    assert seen_transcript[0].text == "what is virtue?"
    assert speaking_as == "socrates"


def test_the_rewritten_query_not_the_raw_message_reaches_the_generator() -> None:
    corpus = _corpus()
    session = PersonaChatSession(
        "socrates", corpus, transcript_store=InMemoryTranscriptStore(), rewriter=_FakeRewriter()
    )

    session.send("what about that?")

    query_seen_by_generator = corpus.generator.calls[0][0]
    assert query_seen_by_generator == "REWRITTEN: what about that?"
    assert query_seen_by_generator != "what about that?"


def test_persona_reply_is_appended_as_plain_answer_text() -> None:
    corpus = _corpus()
    session = PersonaChatSession(
        "socrates", corpus, transcript_store=InMemoryTranscriptStore(), rewriter=_FakeRewriter()
    )

    answer = session.send("what is virtue?")
    transcript = session.transcript()

    assert [t.speaker for t in transcript] == ["user", "socrates"]
    assert transcript[1].text == answer.text


def test_send_returns_the_full_answer_not_a_bare_string() -> None:
    corpus = _corpus()
    session = PersonaChatSession(
        "socrates", corpus, transcript_store=InMemoryTranscriptStore(), rewriter=_FakeRewriter()
    )

    answer = session.send("what is virtue?")

    assert hasattr(answer, "text")
    assert hasattr(answer, "citations")
    assert hasattr(answer, "verification")


def test_multiple_turns_accumulate_in_speaking_order() -> None:
    corpus = _corpus()
    session = PersonaChatSession(
        "socrates", corpus, transcript_store=InMemoryTranscriptStore(), rewriter=_FakeRewriter()
    )

    session.send("first question")
    session.send("second question")

    assert [t.speaker for t in session.transcript()] == ["user", "socrates", "user", "socrates"]


def test_close_clears_the_session_from_the_transcript_store() -> None:
    store = InMemoryTranscriptStore()
    corpus = _corpus()
    session = PersonaChatSession(
        "socrates", corpus, transcript_store=store, rewriter=_FakeRewriter()
    )
    session.send("hello")

    session.close()

    assert store.recent_turns(session.session_id) == []


def test_context_manager_closes_on_exit() -> None:
    store = InMemoryTranscriptStore()
    corpus = _corpus()
    with PersonaChatSession(
        "socrates", corpus, transcript_store=store, rewriter=_FakeRewriter()
    ) as session:
        session.send("hello")
        session_id = session.session_id

    assert store.recent_turns(session_id) == []


def test_separate_sessions_get_separate_session_ids_by_default() -> None:
    corpus = _corpus()
    store = InMemoryTranscriptStore()
    s1 = PersonaChatSession("socrates", corpus, transcript_store=store, rewriter=_FakeRewriter())
    s2 = PersonaChatSession("socrates", corpus, transcript_store=store, rewriter=_FakeRewriter())

    assert s1.session_id != s2.session_id
