"""Protocols for the conversation layer. Nothing in ragcore.types or
ragcore.corpus references ragcore.conversation, so — like ragcore.concepts,
ragcore.hierarchy, and ragcore.claims — this package has no import cycle to
defer around.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ragcore.conversation.types import Turn


@runtime_checkable
class TranscriptStore(Protocol):
    """Persists Turns, scoped by session_id. Session-scoped and ephemeral by
    contract: nothing implementing this protocol may write a Turn into any
    VectorStore/ConceptStore/ClaimStore, and `clear_session` must actually
    free the session's turns, not just mark them inactive — a transcript
    that outlives its session is a memory leak, not a feature, in a
    long-lived server process handling many sessions.
    """

    def append_turn(self, session_id: str, turn: Turn) -> None: ...

    def recent_turns(self, session_id: str, *, limit: int | None = None) -> list[Turn]:
        """Every turn for this session in speaking order, or only the most
        recent `limit` if given. An unknown session_id returns an empty
        list, not an error — a session's first turn always starts from
        nothing."""
        ...

    def clear_session(self, session_id: str) -> None:
        """Frees every turn recorded for this session. Safe to call on an
        already-cleared or unknown session_id (a no-op)."""
        ...


@runtime_checkable
class QueryRewriter(Protocol):
    """Rewrites the next turn's implicit query into a self-contained one a
    retrieval pipeline can act on, using the transcript alone — never the
    corpus, never the persona's own retrieved material (this step runs
    strictly BEFORE any Corpus.retrieve()/ask() call)."""

    model: str

    def rewrite(self, *, transcript: Sequence[Turn], speaking_as: str) -> str:
        """`transcript` ends with the thing prompting this turn — the
        user's latest message in chat mode, the opponent's latest point (or
        a caller-seeded 'topic' turn on the very first turn) in debate mode.
        `speaking_as` is the persona_id (or "user"-facing persona) about to
        respond. Returns one self-contained query string; has no notion of
        "chat" vs. "debate" mode, only "whose turn is it and what's been
        said so far\""""
        ...
