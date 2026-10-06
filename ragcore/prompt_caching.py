"""Shared prompt-caching helpers for Claude-backed components
(docs/ARCHITECTURE.md Phase 5 gap list item 21, prompt-caching half —
model routing, the other half, is ragcore.model_routing).

Anthropic's prompt caching activates on a `cache_control: {"type":
"ephemeral"}` marker attached to a content block, reusing that block's
processed form on a later call whose prefix matches exactly. Marking a
block that's under the API's own minimum cacheable size is harmless — it
simply never activates — so these helpers are safe to apply unconditionally
wherever a caller opts in, without needing to know a given prompt's exact
token count.

Wired into the components where the same content is genuinely resent
unchanged across more than one call: `ClaudeEntityExtractor`/
`ClaudeConceptExtractor`'s gleaning rounds resend the *same* chapter text
as the first user turn on every round of one `extract()` call (see
ragcore.graph.claude_extractor's module docstring), and every
Claude-backed component's system prompt is identical across every call
that component ever makes. Both are real, not speculative, reuse —
`enable_prompt_caching=True` (opt-in; default False keeps a plain string
system prompt and the exact request shape every existing caller already
gets) marks both.
"""

from __future__ import annotations

from typing import Any


def cached_text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}


def cached_system(system_prompt: str) -> list[dict[str, Any]]:
    """A `system=` value with caching enabled — pass this in place of the
    plain string when `enable_prompt_caching=True`."""
    return [cached_text_block(system_prompt)]
