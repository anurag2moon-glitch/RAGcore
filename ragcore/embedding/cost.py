"""UsageTracker — token/request/cost counters for an Embedder instance's
lifetime (Phase 2 gap list item 9: "cost tracking on every provider
call... surfaced on the Embedder/Generator, not just retried-and-
discarded").

Deliberately not part of the Embedder protocol (ragcore/embedding/base.py)
— adding a required attribute there would be a breaking change for every
implementation, including test doubles with no real cost to track.
Instead this is a small building block a concrete implementation attaches
as `self.usage`; VoyageContextEmbedder and OpenAIEmbedder both do.

`price_per_million_tokens_usd` is optional and, where a default is
supplied by a caller, is a best-effort list-price snapshot, not a live
lookup — it will go stale as providers change pricing. Token and request
counts are always exact (ragcore counts every token it sends); the dollar
estimate is the one derived, approximate figure here, and is `None` —
never a guessed number presented as fact — when no price was supplied.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UsageTracker:
    price_per_million_tokens_usd: float | None = None
    total_requests: int = 0
    total_tokens: int = 0
    failed_requests: int = 0

    @property
    def estimated_cost_usd(self) -> float | None:
        if self.price_per_million_tokens_usd is None:
            return None
        return (self.total_tokens / 1_000_000) * self.price_per_million_tokens_usd

    def record_success(self, tokens: int) -> None:
        self.total_requests += 1
        self.total_tokens += tokens

    def record_failure(self) -> None:
        self.failed_requests += 1
