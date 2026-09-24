"""build_lexical_index — the separate pass that populates a LexicalStore
from already-ingested chunks, mirroring build_graph/build_concepts'
attach-after-ingest shape.

Unlike build_graph/build_concepts, this needs no checkpoint table at all:
indexing has no LLM call (nothing to fail expensively or need resuming),
and every write is idempotent by chunk_id (LexicalStore.index() replaces,
same semantics as VectorStore.upsert()), so re-running this in full is
always safe and cheap. Kept as a separate pass anyway, rather than wired
into Corpus.ingest(), purely to preserve one uniform invariant across every
add-on capability in this library: Corpus is never modified by an add-on
layer, full stop — not "except when it happens to be safe."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from ragcore.lexical.base import LexicalStore

if TYPE_CHECKING:
    from ragcore.corpus import Corpus


@dataclass(frozen=True)
class LexicalIndexReport:
    sources_indexed: int
    chunks_indexed: int


def build_lexical_index(
    corpus: Corpus, lexical: LexicalStore, *, source_ids: Sequence[str] | None = None
) -> LexicalIndexReport:
    sources = corpus.sources()
    if source_ids is not None:
        wanted = set(source_ids)
        sources = [s for s in sources if s.source_id in wanted]

    chunks_indexed = 0
    for source in sources:
        chunks = corpus.store.get_by_source(source.source_id)
        lexical.index(chunks)
        chunks_indexed += len(chunks)

    return LexicalIndexReport(sources_indexed=len(sources), chunks_indexed=chunks_indexed)
