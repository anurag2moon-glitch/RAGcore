"""Data types for the conversation layer.

Nothing here is a Chunk, a Concept, or a SourceClaim, and nothing here is
ever handed to a VectorStore/ConceptStore/ClaimStore — a Turn is plain
dialogue text, session-scoped, never written into any persona's corpus. See
ragcore/conversation/__init__.py for the full rationale.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    """One line of a transcript: who said it, and what."""

    speaker: str
    """"user" for the human side of a chat, a persona_id for a persona's own
    reply in either mode, or a caller-defined label (e.g. "topic") used to
    seed a debate transcript before either persona has spoken. Deliberately
    a plain str, not a closed enum — debate speakers are whatever persona
    ids the caller assigns."""

    text: str
    """Plain text — no markdown, no citation markers, no source tags (see
    ragcore.conversation.prompts.persona_system_prompt). Exactly what was
    said, suitable for a transcript a rewriter reads back and, eventually,
    for a text-to-speech layer."""

    created_at: str
    """ISO-8601 UTC timestamp, for display/debugging and stable transcript
    ordering — never used to invalidate anything."""
