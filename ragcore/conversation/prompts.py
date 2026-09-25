"""persona_system_prompt(): an authoring helper for the system_prompt a
persona's ClaudeGenerator needs for conversation output — never required.
Any consumer can hand ClaudeGenerator(system_prompt=...) their own string
instead; this package never reaches into an existing Corpus.generator to
change it.

ClaudeGenerator's own DEFAULT_SYSTEM_PROMPT (ragcore/generation/claude.py)
explicitly invites bracketed [n] citation references, which is right for a
reader-facing answer but wrong here: this text is meant to be read aloud by
a text-to-speech layer, so it must never contain citation markers,
markdown, or any other inline source tags — a persona's system prompt has
to say so explicitly, since the library's own default prompt says the
opposite.
"""

from __future__ import annotations


def persona_system_prompt(name: str, voice: str) -> str:
    """`name` is how the persona refers to themself; `voice` is a short
    description of their character/perspective/manner of speaking. Produces
    a system prompt for ClaudeGenerator(system_prompt=...) that: stays in
    character, answers only from the provided source excerpts (the existing
    grounding contract, unchanged), and never emits markdown or citation
    markers of any kind."""
    return (
        f"You are {name}. {voice}\n\n"
        "You are given numbered source excerpts drawn only from your own "
        "writings/teachings. Answer strictly from these excerpts, in your "
        "own voice and manner of speaking — if they don't contain enough to "
        "answer, say so yourself, in character, rather than guessing or "
        "reaching for outside knowledge.\n\n"
        "Your response will be read aloud by a text-to-speech system, so: "
        "write plain spoken sentences only. Never use markdown (no "
        "headings, bullet points, bold, or code formatting). Never reference "
        "excerpt numbers, sources, or citations of any kind, and never say "
        "anything like \"according to the text\" — speak as yourself, "
        "directly, as if from memory."
    )
