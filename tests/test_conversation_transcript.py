"""InMemoryTranscriptStore: append/recent/limit/clear, unknown-session
defaults, and per-session isolation of a shared store instance.
"""

from __future__ import annotations

import threading

from ragcore.conversation.transcript import InMemoryTranscriptStore
from ragcore.conversation.types import Turn


def _turn(speaker: str, text: str) -> Turn:
    return Turn(speaker=speaker, text=text, created_at="now")


def test_recent_turns_on_an_unknown_session_returns_empty_list() -> None:
    store = InMemoryTranscriptStore()
    assert store.recent_turns("nope") == []


def test_append_then_recent_turns_round_trips_in_order() -> None:
    store = InMemoryTranscriptStore()
    store.append_turn("s1", _turn("user", "hello"))
    store.append_turn("s1", _turn("socrates", "greetings"))

    turns = store.recent_turns("s1")
    assert [t.speaker for t in turns] == ["user", "socrates"]
    assert [t.text for t in turns] == ["hello", "greetings"]


def test_limit_returns_only_the_most_recent_n_turns() -> None:
    store = InMemoryTranscriptStore()
    for i in range(5):
        store.append_turn("s1", _turn("user", f"msg{i}"))

    turns = store.recent_turns("s1", limit=2)
    assert [t.text for t in turns] == ["msg3", "msg4"]


def test_sessions_are_isolated_from_each_other() -> None:
    store = InMemoryTranscriptStore()
    store.append_turn("s1", _turn("user", "for s1"))
    store.append_turn("s2", _turn("user", "for s2"))

    assert [t.text for t in store.recent_turns("s1")] == ["for s1"]
    assert [t.text for t in store.recent_turns("s2")] == ["for s2"]


def test_clear_session_frees_its_turns_and_is_safe_to_repeat() -> None:
    store = InMemoryTranscriptStore()
    store.append_turn("s1", _turn("user", "hello"))

    store.clear_session("s1")
    assert store.recent_turns("s1") == []

    store.clear_session("s1")  # no-op, must not raise
    store.clear_session("never-existed")  # no-op, must not raise


def test_concurrent_sessions_do_not_corrupt_the_shared_dict() -> None:
    store = InMemoryTranscriptStore()
    n_sessions = 20
    n_turns_each = 25

    def worker(session_id: str) -> None:
        for i in range(n_turns_each):
            store.append_turn(session_id, _turn("user", f"{session_id}-{i}"))

    threads = [
        threading.Thread(target=worker, args=(f"session-{i}",)) for i in range(n_sessions)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i in range(n_sessions):
        turns = store.recent_turns(f"session-{i}")
        assert len(turns) == n_turns_each
        assert [t.text for t in turns] == [f"session-{i}-{j}" for j in range(n_turns_each)]
