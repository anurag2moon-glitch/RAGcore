"""InMemoryTranscriptStore — default TranscriptStore: one process-local
dict, session-scoped, never persisted anywhere.

Guarded by a threading.Lock around dict-level mutation (creating a new
session's list, appending to it, clearing it) — the same concurrency
discipline every SQLite-backed store in this codebase already follows, and
directly relevant here: a server handling many concurrent chat/debate
sessions will call into one shared TranscriptStore instance from many
request threads at once, each touching a *different* session_id, so the
dict itself (not any one session's list) is the shared mutable state that
needs protecting.

This is the reference implementation, correct for a single-process
deployment. A horizontally-scaled deployment (many server instances behind
a load balancer) needs a shared backend — Redis, a database row per
session — keyed the same way (session_id -> ordered Turns); that's a
drop-in TranscriptStore implementation, not a change to this protocol or to
anything in ragcore.conversation.session, which only ever depends on the
Protocol.
"""

from __future__ import annotations

import threading

from ragcore.conversation.types import Turn


class InMemoryTranscriptStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, list[Turn]] = {}

    def append_turn(self, session_id: str, turn: Turn) -> None:
        with self._lock:
            self._sessions.setdefault(session_id, []).append(turn)

    def recent_turns(self, session_id: str, *, limit: int | None = None) -> list[Turn]:
        with self._lock:
            turns = list(self._sessions.get(session_id, ()))
        return turns if limit is None else turns[-limit:]

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
