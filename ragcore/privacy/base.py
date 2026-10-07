"""PiiDetector protocol: a swappable, zero-call (no LLM, no network by
default) scan of raw text for personally identifiable information.

Kept deliberately simple — pattern-based detection of a handful of
well-known PII shapes, not a full NER model. A consumer who needs
higher-recall detection (e.g. named-entity-based PII like addresses or
person names) implements this same Protocol with their own backend
(regex, a hosted PII-detection API, a local NER model); nothing else in
ragcore needs to change for that swap, same as every other Protocol in
this codebase.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragcore.privacy.types import PiiSpan


@runtime_checkable
class PiiDetector(Protocol):
    def detect(self, text: str) -> list[PiiSpan]:
        """Every PII span found in `text`, in any order. Pure function of
        `text` alone — a detector has no chunk/corpus context, so
        ragcore.privacy.pipeline.scan_corpus is what anchors a PiiSpan
        back to the chunk_id/source_id it came from."""
        ...
