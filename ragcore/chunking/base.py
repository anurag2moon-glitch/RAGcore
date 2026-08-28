"""Chunker protocol: a Document -> groups of Chunks.

See ragcore.types.ChunkGroup for why chunkers return groups rather than a
flat chunk list: a group is the contextualization unit an Embedder embeds
jointly. Chunk.vector is unset at this stage — the Embedder fills it in.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragcore.types import ChunkGroup, Document


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, doc: Document) -> list[ChunkGroup]: ...
