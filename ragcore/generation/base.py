"""Generator protocol: query + retrieved context -> a synthesized answer.

Same shape as Embedder/VectorStore/Chunker — a swappable Protocol, not a
concrete dependency. Corpus.ask() composes Corpus.retrieve() with whatever
Generator a consumer configures; ragcore ships one implementation
(ClaudeGenerator), but nothing in Corpus or the retrieval path depends on
it existing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

from ragcore.types import Retrieved

if TYPE_CHECKING:
    # Deferred: ragcore.verification imports ragcore.generation.base (for
    # Answer/Generator), so importing Verification back here at module level
    # would cycle. Verification is only ever used here as a type hint.
    from ragcore.verification.types import Verification


@dataclass(frozen=True)
class Answer:
    """A synthesized answer, grounded in the chunks it was generated from."""

    text: str
    citations: list[Retrieved]
    """The exact Retrieved chunks passed as context — i.e. what the answer
    was (asked to be) grounded in, not necessarily what it actually cited
    inline. Lets a caller show "sources" alongside the answer. A citation
    may be a synthesized community summary rather than a real passage —
    check `chunk.meta.extra.get("ragcore_kind") == "community_summary"`
    (see ragcore.concepts.communities) before presenting one as a quoted
    source."""

    verification: Verification | None = None
    """Set only when a ragcore.verification.claude.VerifyingGenerator wraps
    the Generator that produced this Answer — None otherwise, including for
    every plain ClaudeGenerator/FakeGenerator answer. Per-claim groundedness
    against `citations`; see ragcore.verification."""


@runtime_checkable
class Generator(Protocol):
    model: str

    def generate(self, query: str, context: Sequence[Retrieved]) -> Answer:
        """Synthesize an answer to `query` using only `context`."""
        ...
