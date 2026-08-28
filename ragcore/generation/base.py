"""Generator protocol: query + retrieved context -> a synthesized answer.

Same shape as Embedder/VectorStore/Chunker — a swappable Protocol, not a
concrete dependency. Corpus.ask() composes Corpus.retrieve() with whatever
Generator a consumer configures; ragcore ships one implementation
(ClaudeGenerator), but nothing in Corpus or the retrieval path depends on
it existing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from ragcore.types import Retrieved


@dataclass(frozen=True)
class Answer:
    """A synthesized answer, grounded in the chunks it was generated from."""

    text: str
    citations: list[Retrieved]
    """The exact Retrieved chunks passed as context — i.e. what the answer
    was (asked to be) grounded in, not necessarily what it actually cited
    inline. Lets a caller show "sources" alongside the answer."""


@runtime_checkable
class Generator(Protocol):
    model: str

    def generate(self, query: str, context: Sequence[Retrieved]) -> Answer:
        """Synthesize an answer to `query` using only `context`."""
        ...
