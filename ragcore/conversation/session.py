"""PersonaChatSession — the base layer's conversation entrypoint: append/
read Turns from a TranscriptStore, rewrite via a QueryRewriter, then call
the EXISTING, unmodified `Corpus.ask(query, ...)` entrypoint. Touches no
retrieval/generation/verification internals, and never writes a Turn into
any corpus/store — see the module docstring in ragcore.conversation.base
for the isolation contract this relies on.

`send()` answers with the full `Answer` (text + citations + verification, if
the persona's own Corpus has a VerifyingGenerator wired) — never a bare
string. Discarding citations/verification the existing pipeline already
computed would throw away real information for no reason; only the
TRANSCRIPT is text-only (`answer.text`), matching the text-to-speech-ready
output requirement and keeping the next rewrite step free of citation
clutter.

Multi-party sessions (e.g. two personas debating each other) are a
domain-specific product feature, not a generic base-layer primitive — see
CLAUDE.md's "Conversation layer" section — and are built as a separate
consuming layer on top of TranscriptStore/QueryRewriter/PersonaChatSession,
not inside ragcore.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ragcore.conversation.base import QueryRewriter, TranscriptStore
from ragcore.conversation.types import Turn

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
