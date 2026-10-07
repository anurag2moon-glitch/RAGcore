"""Data types for ragcore.privacy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PiiSpan:
    """One PII hit within a single text, as a PiiDetector reports it —
    chunk-agnostic, since a detector only ever sees raw text."""

    pii_type: str
    """Open vocabulary (e.g. "email", "phone", "ssn", "credit_card") —
    not an enum, since a custom PiiDetector may detect domain-specific
    categories a built-in one never would."""

    start: int
    end: int


@dataclass(frozen=True)
class PiiMatch:
    """One PiiSpan anchored to the chunk/source it was found in.

    Deliberately carries no copy of the matched substring itself —
    `start`/`end` are enough for a caller holding the real `Chunk.text`
    to look at the flagged span deliberately, and a scan report is
    exactly the kind of object that tends to get logged, dashboarded, or
    checked into a CI artifact; it should never become a second place
    where the very PII it's flagging ends up persisted in plaintext."""

    pii_type: str
    chunk_id: str
    source_id: str
    start: int
    end: int


@dataclass(frozen=True)
class PiiScanReport:
    chunks_scanned: int
    matches: tuple[PiiMatch, ...]

    def counts_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in self.matches:
            counts[m.pii_type] = counts.get(m.pii_type, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks_scanned": self.chunks_scanned,
            "counts_by_type": self.counts_by_type(),
            "matches": [
                {
                    "pii_type": m.pii_type,
                    "chunk_id": m.chunk_id,
                    "source_id": m.source_id,
                    "start": m.start,
                    "end": m.end,
                }
                for m in self.matches
            ],
        }
