"""Verifier protocol: checks an already-generated Answer's claims against
the context it was generated from.

A genuinely separate pass from generation — a Verifier is handed the
finished Answer and its context, after the fact, never involved in
producing the answer text itself. See ragcore.verification.claude for the
default implementation and VerifyingGenerator, the Generator decorator that
wires a Verifier in.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragcore.generation.base import Answer
from ragcore.verification.types import Verification


@runtime_checkable
class Verifier(Protocol):
    model: str

    def verify(self, query: str, answer: Answer) -> Verification:
        """Checks every claim in `answer.text` against `answer.citations`
        (the passages/summaries the answer was generated from) and returns
        a per-claim verdict."""
        ...
