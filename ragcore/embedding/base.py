"""Embedder protocol: text -> vectors.

Shaped around contextualized embedding (groups of chunks embedded jointly)
because that is the superset of what any embedder can do — a backend
without contextualization support just embeds each chunk in its group
independently. Shaping the protocol around the less capable case would make
it impossible for a contextualizing backend (e.g. voyage-context-4) to ever
express what it does through this interface.
"""

from __future__ import annotations

from typing import Literal, Protocol, Sequence, runtime_checkable

from ragcore.types import Vector


@runtime_checkable
class Embedder(Protocol):
    name: str
    """Backend identifier, e.g. "voyage-context-4". Stored on ChunkMeta.extra
    or the store's schema so a corpus can detect an embedder swap."""

    dim: int
    """Output vector dimensionality. A VectorStore is created against a
    fixed dim; swapping embedders with a different dim requires a new
    store (or a re-embed into the same dim)."""

    def embed_groups(
        self,
        groups: Sequence[Sequence[str]],
        *,
        input_type: Literal["document", "query", None] = "document",
    ) -> list[list[Vector]]:
        """Embed each group's texts, contextualized within that group.

        `groups[i]` is the ordered chunk texts of one ChunkGroup (e.g. one
        chapter). Returns one list of vectors per group, same length and
        order as the input group. Groups are independent of each other —
        no cross-group contextualization.
        """
        ...

    def embed_query(self, text: str) -> Vector:
        """Embed a single query string for retrieval."""
        ...
