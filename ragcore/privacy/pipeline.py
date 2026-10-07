"""scan_corpus — the top-level entry point: runs a PiiDetector over every
chunk of an already-ingested Corpus (or a `source_ids`-restricted subset
of it) and returns a PiiScanReport.

A separate, standalone pass over an already-ingested corpus, not a hook
inside Corpus.ingest() — same shape as ragcore.eval.run_eval and every
build_*() pass in this codebase (build_graph, build_concepts, ...):
ingest() stays untouched, and a consumer who never calls scan_corpus()
pays nothing for this layer existing. To scope a scan to one tenant with
ragcore.tenancy.TenantScopedCorpus, pass
`source_ids=[s.source_id for s in scoped.sources()]` and `scoped.corpus`
as the corpus — scan_corpus takes a plain Corpus, not a TenantScopedCorpus,
since all it needs is `.sources()`/`.store.get_by_source()`, already
correctly tenant-scoped by the caller's own `source_ids` filter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from ragcore.privacy.base import PiiDetector
from ragcore.privacy.types import PiiMatch, PiiScanReport

if TYPE_CHECKING:
    from ragcore.corpus import Corpus


def scan_corpus(
    corpus: Corpus, detector: PiiDetector, *, source_ids: Sequence[str] | None = None
) -> PiiScanReport:
    sources = corpus.sources()
    if source_ids is not None:
        wanted = set(source_ids)
        sources = [s for s in sources if s.source_id in wanted]

    matches: list[PiiMatch] = []
    chunks_scanned = 0
    for source in sources:
        for chunk in corpus.store.get_by_source(source.source_id):
            chunks_scanned += 1
            for span in detector.detect(chunk.text):
                matches.append(
                    PiiMatch(
                        pii_type=span.pii_type,
                        chunk_id=chunk.chunk_id,
                        source_id=chunk.meta.source_id,
                        start=span.start,
                        end=span.end,
                    )
                )

    return PiiScanReport(chunks_scanned=chunks_scanned, matches=tuple(matches))
