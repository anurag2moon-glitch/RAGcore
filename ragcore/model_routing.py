"""DEFAULT_ROUTING — a shared cheap/strong model-tier default for every
Claude-backed component in this codebase (docs/ARCHITECTURE.md Phase 5
gap list item 21: "model routing (cheap model for rewriting/extraction,
strong model for synthesis)... every Claude-backed component takes its
own model param but nothing centrally routes cheap-vs-strong").

Before this, every Claude-backed component defaulted its own `model`
constructor parameter to `"claude-opus-5"` independently — correct in
that each one could always be overridden, but with no shared convention
for *which* components should default cheap vs. strong, and no one place
that convention lived or could be read as a whole.

`DEFAULT_ROUTING.strong` is unchanged from every synthesis-facing
component's prior hardcoded default — `ClaudeGenerator` (final
user-facing answer synthesis), `ClaudeVerifier` (the faithfulness check
that polices what the strong model said, so it needs comparable
judgment), and `ClaudeCommunitySummarizer` (cross-source synthesized
prose a user may read directly, not an intermediate artifact) keep
exactly the model they defaulted to before this change — a no-op for
that tier.

`DEFAULT_ROUTING.cheap` is a genuine, stated default change for every
other Claude-backed component: extraction (`ClaudeEntityExtractor`,
`ClaudeConceptExtractor`, `ClaimExtractor`, `ClaimLinker`), rewriting
(`HistoryAwareQueryRewriter`, `ClaudeContextWriter`),
routing/classification/judging (`ClaudeQueryRouter`,
`ClaudeQueryExpander`, `ClaudeSufficiencyJudge`, `ClaudeConceptMapper`),
and intermediate (non-final-answer) summarization
(`ClaudeHierarchySummarizer`). These are high-volume, auxiliary calls
where real quality matters far less than cost/latency, relative to the
one call a user's actual answer depends on — this is the behavior the
gap item asks for, not merely an available knob nobody would reach for.

Every component still takes its own `model=...` constructor argument and
can be overridden individually, exactly as before; this module only
changes what each one defaults to when the caller doesn't.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRouting:
    cheap: str = "claude-haiku-4-5"
    strong: str = "claude-opus-5"


DEFAULT_ROUTING = ModelRouting()
