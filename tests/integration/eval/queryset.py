"""Query sets: ground-truth (query, relevant chunk ids) pairs the eval
report runs against. Built from the synthetic library's planted needles
(see tests/integration/synth/library.py) — the relevant set for each query
is computed by scanning real ingested chunk text for the needle's unique
marker token, not assumed from authoring-time offsets, so it's correct
regardless of how the chunker happened to pack paragraphs around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tests.integration.synth.library import Needle, find_chunks_with_marker


@dataclass(frozen=True)
class QuerySpec:
    label: str
    query: str
    relevant: frozenset[str]


def needle_queryset(corpus, needles: Sequence[Needle]) -> list[QuerySpec]:
    specs: list[QuerySpec] = []
    for n in needles:
        relevant = frozenset(find_chunks_with_marker(corpus, n.marker))
        if not relevant:
            continue  # book not ingested in this run (e.g. RAGCORE_IT_SCALE=quick)
        specs.append(QuerySpec(label=f"{n.book_key}:{n.marker}", query=n.query, relevant=relevant))
    return specs
