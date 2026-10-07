"""InjectionDetector protocol: a swappable, zero-call (no LLM, no network
by default) scan of raw text for likely prompt-injection content — text
that tries to redirect an LLM reading it as retrieved context (e.g.
"ignore the above instructions and instead...", embedded inside a
document that later gets retrieved and handed to a Generator).

Kept deliberately heuristic, the same honesty ragcore.privacy.base states
for PII detection: this catches commonly-seen injection phrasings, not
every possible one, and some false positives are expected (a document
that quotes or discusses prompt injection as a topic will trip these
patterns too). A consumer with stricter requirements implements this same
Protocol with a stronger backend (an LLM-based classifier, a hosted
injection-detection service).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragcore.injection.types import InjectionSpan


@runtime_checkable
class InjectionDetector(Protocol):
    def detect(self, text: str) -> list[InjectionSpan]:
        """Every suspected injection span found in `text`, in any order.
        Pure function of `text` alone."""
        ...
