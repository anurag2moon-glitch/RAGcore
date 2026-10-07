"""RegexInjectionDetector — the default InjectionDetector: a handful of
commonly-seen prompt-injection phrasings, via stdlib `re` (zero new
dependency). See InjectionDetector's module docstring for the honesty
disclaimer: heuristic, not exhaustive, and no false positives/negatives
have been systematically measured here. Every pattern is case-insensitive.
"""

from __future__ import annotations

import re

from ragcore.injection.types import InjectionSpan

_PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore_instructions": re.compile(
        r"\bignore\s+(?:all\s+|the\s+)?(?:previous|above|prior)\s+"
        r"(?:instructions?|prompts?|directions?)\b",
        re.IGNORECASE,
    ),
    "disregard_instructions": re.compile(
        r"\bdisregard\s+(?:all\s+|the\s+)?(?:previous|above|prior)\s+"
        r"(?:instructions?|prompts?|directions?)\b",
        re.IGNORECASE,
    ),
    "role_override": re.compile(
        r"\byou\s+are\s+now\s+(?:a|an)\b|"
        r"\bact\s+as\s+(?:if\s+you\s+are|a|an)\b|"
        r"\bnew\s+(?:system\s+)?instructions?:",
        re.IGNORECASE,
    ),
    "reveal_system_prompt": re.compile(
        r"\b(?:reveal|print|repeat|show)\s+(?:your\s+|the\s+)?"
        r"(?:system\s+prompt|instructions?)\b",
        re.IGNORECASE,
    ),
    "end_of_context_marker": re.compile(
        r"\[?/?(?:end|stop)\s+of\s+(?:context|document|instructions?)\]?",
        re.IGNORECASE,
    ),
}


class RegexInjectionDetector:
    """Default InjectionDetector. See module docstring."""

    def detect(self, text: str) -> list[InjectionSpan]:
        spans: list[InjectionSpan] = []
        for pattern_name, regex in _PATTERNS.items():
            for m in regex.finditer(text):
                spans.append(InjectionSpan(pattern=pattern_name, start=m.start(), end=m.end()))
        return spans
