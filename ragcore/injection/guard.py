"""GuardedGenerator — a Generator decorator that screens retrieved context
for likely prompt-injection content before generation, the same seam
ragcore.verification.claude.VerifyingGenerator already uses for post-hoc
groundedness checking:

    corpus.generator = GuardedGenerator(corpus.generator, RegexInjectionDetector())

A consumer who never wires this in sees no change at all.

Deliberately never rewrites `Chunk.text` — unlike
ragcore.contextual.pipeline's blurb (which is careful to stay out of
`Chunk.text` specifically so citations and verification see the real
passage), silently editing out a flagged span here would leave two
inconsistent versions of "the source passage" in circulation: the
(edited) text the Generator sees, and the (original) text a citation
lookup via `Corpus.get(chunk_id)` would still return — which could
re-surface the very content this exists to screen. Instead:
- `mode="flag"` (default): every chunk's `meta.extra["injection_flagged"]`
  is set (True/False), but every chunk still reaches the inner Generator
  unchanged — the consuming application decides what "flagged" should
  mean for its own system prompt / UI, since that's inherently
  product-specific.
- `mode="exclude"`: a flagged chunk is dropped from the context handed to
  the inner Generator entirely (never partially edited), so flagged
  content is excluded from this answer, not laundered into using
  a different-looking version of the same text.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Sequence

from ragcore.generation.base import Answer, Generator
from ragcore.injection.base import InjectionDetector
from ragcore.types import Retrieved


class GuardedGenerator:
    def __init__(
        self,
        inner: Generator,
        detector: InjectionDetector,
        *,
        mode: Literal["flag", "exclude"] = "flag",
    ) -> None:
        self._inner = inner
        self._detector = detector
        self._mode = mode
        self.model = inner.model

    def generate(self, query: str, context: Sequence[Retrieved]) -> Answer:
        guarded: list[Retrieved] = []
        for r in context:
            flagged = bool(self._detector.detect(r.chunk.text))
            if flagged and self._mode == "exclude":
                continue
            if flagged:
                stamped = dataclasses.replace(
                    r.chunk,
                    meta=dataclasses.replace(
                        r.chunk.meta,
                        extra={**r.chunk.meta.extra, "injection_flagged": True},
                    ),
                )
                guarded.append(dataclasses.replace(r, chunk=stamped))
            else:
                guarded.append(r)
        return self._inner.generate(query, guarded)
