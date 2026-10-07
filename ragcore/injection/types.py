"""Data types for ragcore.injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InjectionSpan:
    """One suspected prompt-injection span within a single text, as an
    InjectionDetector reports it — chunk-agnostic, since a detector only
    ever sees raw text."""

    pattern: str
    """Which heuristic matched (e.g. "ignore_instructions",
    "role_override") — open vocabulary, not an enum, since a custom
    InjectionDetector may define its own categories."""

    start: int
    end: int


@dataclass(frozen=True)
class InjectionMatch:
    """One InjectionSpan anchored to the chunk/source it was found in."""

    pattern: str
    chunk_id: str
    source_id: str
    start: int
    end: int


@dataclass(frozen=True)
class InjectionScanReport:
    chunks_scanned: int
    matches: tuple[InjectionMatch, ...]

    def counts_by_pattern(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in self.matches:
            counts[m.pattern] = counts.get(m.pattern, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks_scanned": self.chunks_scanned,
            "counts_by_pattern": self.counts_by_pattern(),
            "matches": [
                {
                    "pattern": m.pattern,
                    "chunk_id": m.chunk_id,
                    "source_id": m.source_id,
                    "start": m.start,
                    "end": m.end,
                }
                for m in self.matches
            ],
        }
