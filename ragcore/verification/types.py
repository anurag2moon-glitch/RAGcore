"""Data types for post-generation citation verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ClaimStatus = Literal["supported", "supported_by_synthesis", "unsupported"]


@dataclass(frozen=True)
class ClaimCheck:
    """One atomic claim from an Answer's text, checked against the passages
    it was generated from."""

    claim: str
    """A verbatim span of Answer.text, not a paraphrase — so a caller can
    locate it in the original answer (e.g. for highlighting)."""

    status: ClaimStatus
    support: tuple[str, ...]
    """chunk_ids (see ragcore.types.Chunk.chunk_id) of context items the
    verifier judged this claim is supported by — real passages, synthetic
    community-summary chunks, or both."""

    note: str | None = None
    """Optional short explanation, mainly useful when status is
    "unsupported" — why the verifier couldn't ground this claim."""


@dataclass(frozen=True)
class Verification:
    """Per-claim groundedness check for one Answer. See
    ragcore.verification.claude.VerifyingGenerator, which attaches this to
    Answer.verification — never computed as part of the generation call
    itself."""

    checks: tuple[ClaimCheck, ...]
    model: str

    @property
    def unsupported(self) -> tuple[ClaimCheck, ...]:
        return tuple(c for c in self.checks if c.status == "unsupported")

    @property
    def supported_ratio(self) -> float:
        if not self.checks:
            return 1.0
        supported = sum(1 for c in self.checks if c.status != "unsupported")
        return supported / len(self.checks)
