"""scan_corpus — runs an InjectionDetector over every chunk of an
already-ingested Corpus (or a `source_ids`-restricted subset) and returns
an InjectionScanReport. Same shape and same "standalone pass, ingest()
untouched" rationale as ragcore.privacy.pipeline.scan_corpus — see that
module's docstring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from ragcore.injection.base import InjectionDetector
from ragcore.injection.types import InjectionMatch, InjectionScanReport

if TYPE_CHECKING:
    from ragcore.corpus import Corpus


def scan_corpus(
    corpus: Corpus, detector: InjectionDetector, *, source_ids: Sequence[str] | None = None
) -> InjectionScanReport:
    sources = corpus.sources()
    if source_ids is not None:
        wanted = set(source_ids)
        sources = [s for s in sources if s.source_id in wanted]

    matches: list[InjectionMatch] = []
    chunks_scanned = 0
    for source in sources:
        for chunk in corpus.store.get_by_source(source.source_id):
            chunks_scanned += 1
            for span in detector.detect(chunk.text):
                matches.append(
                    InjectionMatch(
                        pattern=span.pattern,
                        chunk_id=chunk.chunk_id,
                        source_id=chunk.meta.source_id,
                        start=span.start,
                        end=span.end,
                    )
                )

    return InjectionScanReport(chunks_scanned=chunks_scanned, matches=tuple(matches))
