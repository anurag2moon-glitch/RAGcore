"""PersonaChatSession and DebateSession — the two entrypoints of the
conversation layer, both built the same way: append/read Turns from a
TranscriptStore, rewrite via a QueryRewriter, then call the EXISTING,
unmodified `Corpus.ask(query, ...)` entrypoint. Neither class touches
retrieval/generation/verification internals, and neither ever writes a Turn
into any corpus/store — see the module docstring in
ragcore.conversation.base for the isolation contract this relies on.

PersonaChatSession answers `send()`/`take_turn()` with the full `Answer`
(text + citations + verification, if the persona's own Corpus has a
VerifyingGenerator wired) — never a bare string. Discarding
citations/verification the existing pipeline already computed would throw
away real information for no reason; only the TRANSCRIPT is text-only
(`answer.text`), matching the text-to-speech-ready output requirement and
keeping the next rewrite step free of citation clutter.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping

from ragcore.conversation.base import QueryRewriter, TranscriptStore
from ragcore.conversation.types import Turn
from ragcore.errors import SessionError

if TYPE_CHECKING:
    from ragcore.corpus import Corpus
    from ragcore.generation.base import Answer
    from ragcore.types import Filter


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_session_id() -> str:
    return uuid.uuid4().hex


class PersonaChatSession:
    """A user talking with one persona. `corpus` is that persona's own,
    already-configured Corpus — its retriever/generator (voice, grounding,
    verification) are exactly whatever the caller set up beforehand; this
    class changes none of it."""

    def __init__(
        self,
        persona_id: str,
        corpus: Corpus,
        *,
        transcript_store: TranscriptStore,
        rewriter: QueryRewriter,
        session_id: str | None = None,
        max_history_turns: int = 12,
    ) -> None:
        self.persona_id = persona_id
        self.session_id = session_id or _new_session_id()
        self._corpus = corpus
        self._transcript_store = transcript_store
        self._rewriter = rewriter
        self._max_history_turns = max_history_turns

    def send(self, user_message: str, *, k: int = 10, filters: Filter | None = None) -> Answer:
        self._transcript_store.append_turn(
            self.session_id, Turn(speaker="user", text=user_message, created_at=_now())
        )
        history = self._transcript_store.recent_turns(
            self.session_id, limit=self._max_history_turns
        )
        rewritten = self._rewriter.rewrite(transcript=history, speaking_as=self.persona_id)

        # The rewritten query is a plain string handed to the existing,
        # unmodified retrieval/generation entrypoint — this is the entire
        # integration point with the rest of ragcore.
        answer = self._corpus.ask(rewritten, k=k, filters=filters)

        self._transcript_store.append_turn(
            self.session_id, Turn(speaker=self.persona_id, text=answer.text, created_at=_now())
        )
        return answer

    def transcript(self) -> list[Turn]:
        return self._transcript_store.recent_turns(self.session_id)

    def close(self) -> None:
        self._transcript_store.clear_session(self.session_id)

    def __enter__(self) -> PersonaChatSession:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class DebateSession:
    """Two (or more) personas addressing the same transcript, each speaking
    only through their own Corpus. `personas` maps persona_id -> that
    persona's own, already-configured Corpus — the only way a persona's
    turn ever reaches a Corpus is `personas[speaker_persona_id].ask(...)`,
    so there is no code path here through which one persona's generation
    step could be handed another persona's Corpus/retriever/generator/store.
    """

    def __init__(
        self,
        personas: Mapping[str, Corpus],
        *,
        transcript_store: TranscriptStore,
        rewriter: QueryRewriter,
        session_id: str | None = None,
        max_history_turns: int = 12,
    ) -> None:
        seen_ids: set[int] = set()
        for persona_id, corpus in personas.items():
            if id(corpus) in seen_ids:
                raise SessionError(
                    f"two personas in this DebateSession share the same Corpus instance "
                    f"(persona {persona_id!r} points at a Corpus already used by another "
                    "persona) — each persona must have its own, separate Corpus, or "
                    "corpus-global passes (detect_communities, link_claims) would "
                    "silently cross-contaminate their concepts/claims"
                )
            seen_ids.add(id(corpus))

        self._personas = dict(personas)
        self.session_id = session_id or _new_session_id()
        self._transcript_store = transcript_store
        self._rewriter = rewriter
        self._max_history_turns = max_history_turns

    def seed_topic(self, text: str, *, speaker: str = "topic") -> None:
        """Appends the framing turn a debate needs before either persona has
        spoken — take_turn()'s rewriter has nothing to react to on an empty
        transcript, so a real debate always starts with this (or a first
        take_turn() would just echo an empty query)."""
        self._transcript_store.append_turn(
            self.session_id, Turn(speaker=speaker, text=text, created_at=_now())
        )

    def take_turn(
        self, speaker_persona_id: str, *, k: int = 10, filters: Filter | None = None
    ) -> Answer:
        if speaker_persona_id not in self._personas:
            raise SessionError(
                f"unknown persona_id {speaker_persona_id!r}; this DebateSession "
                f"only knows {sorted(self._personas)}"
            )

        history = self._transcript_store.recent_turns(
            self.session_id, limit=self._max_history_turns
        )
        rewritten = self._rewriter.rewrite(transcript=history, speaking_as=speaker_persona_id)

        # Exactly one persona's own Corpus is ever touched per turn — the
        # opponent's Corpus/retriever/generator/store is never referenced.
        corpus = self._personas[speaker_persona_id]
        answer = corpus.ask(rewritten, k=k, filters=filters)

        self._transcript_store.append_turn(
            self.session_id,
            Turn(speaker=speaker_persona_id, text=answer.text, created_at=_now()),
        )
        return answer

    def transcript(self) -> list[Turn]:
        return self._transcript_store.recent_turns(self.session_id)

    def close(self) -> None:
        self._transcript_store.clear_session(self.session_id)

    def __enter__(self) -> DebateSession:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
