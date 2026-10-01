"""group_fingerprint — the content fingerprint Corpus.ingest() checkpoints
a ChunkGroup against when a checkpoint_store is configured.

Not `sorted(chunk_ids)` alone, unlike ragcore.concepts/.contextual's
precedent: `Chunk.chunk_id` here is positional
(`f"{source_id}:{chunk_index}"`, see ragcore.chunking.structural), not
content-addressed, so two different chunker configurations over the same
file can produce the identical chunk_id sequence with different actual
text at each index. Hashing the chunk texts themselves (rather than their
ids) is what makes a chunker-config change, an embedder swap, or a
different `metadata=` argument to ingest() correctly invalidate every
checkpoint they affect, instead of silently resuming onto stale content.
See docs/adr/0002-ingest-checkpointing-opt-in.md.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from ragcore.types import Chunk


def group_fingerprint(
    *, embedder_name: str, metadata: dict[str, Any] | None, chunks: Sequence[Chunk]
) -> str:
    text_digest = hashlib.sha256(
        "\x1f".join(c.text for c in chunks).encode("utf-8")
    ).hexdigest()
    metadata_blob = json.dumps(metadata or {}, sort_keys=True, default=str)
    payload = "|".join([embedder_name, metadata_blob, str(len(chunks)), text_digest])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
