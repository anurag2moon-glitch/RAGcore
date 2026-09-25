"""Conversation layer: session-scoped transcripts and context-aware query
rewriting, attached in front of an existing Corpus without any change to
retrieval, generation, or verification.

`PersonaChatSession` is the one entrypoint here: a user talking with one
persona, built by appending/reading Turns from a TranscriptStore, rewriting
via a QueryRewriter, then calling the unmodified Corpus.ask(). A multi-party
session (e.g. two personas debating each other) is a domain-specific product
feature, not a generic base-layer primitive, and is built as a separate
consuming layer on top of the primitives exported here — see CLAUDE.md's
"Conversation layer" section for the reasoning.

The transcript is never written into any persona's corpus, concept graph,
or claim store — it lives only in a `TranscriptStore` (in-memory by
default) for the lifetime of one session, and `close()`/`clear_session()`
frees it explicitly.

    from ragcore.conversation import (
        Turn,                                          # transcript entry
        TranscriptStore, QueryRewriter,                 # protocols
        InMemoryTranscriptStore, HistoryAwareQueryRewriter,  # defaults
        persona_system_prompt,                          # authoring helper
        PersonaChatSession,                              # entrypoint
    )
"""

from ragcore.conversation.base import QueryRewriter, TranscriptStore
from ragcore.conversation.prompts import persona_system_prompt
from ragcore.conversation.rewriting import HistoryAwareQueryRewriter
from ragcore.conversation.session import PersonaChatSession
from ragcore.conversation.transcript import InMemoryTranscriptStore
from ragcore.conversation.types import Turn

__all__ = [
    "Turn",
    "TranscriptStore",
    "QueryRewriter",
    "InMemoryTranscriptStore",
    "HistoryAwareQueryRewriter",
    "persona_system_prompt",
    "PersonaChatSession",
]
