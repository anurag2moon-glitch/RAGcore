"""RegexPiiDetector — the default PiiDetector: email, phone (US-style),
SSN (US-style), and credit-card-number shapes, all via stdlib `re` (zero
new dependency).

Pattern-based detection is inherently approximate — see each pattern's
own comment for its known false-positive/false-negative shape. This is
a reasonable default for "does this corpus likely contain PII at all,"
not a compliance-grade guarantee; a consumer with stricter requirements
should swap in a detector backed by a dedicated PII-detection service.
"""

from __future__ import annotations

import re

from ragcore.privacy.types import PiiSpan

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

_PHONE_RE = re.compile(
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)
"""US-style 10-digit phone numbers, optionally with a +1 country code
and common separators. Will also match some non-phone 10-digit
sequences (e.g. certain invoice/order numbers) — a known false-positive
shape inherent to the format being ambiguous on its own."""

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
"""Only the hyphenated US SSN format (###-##-####) — deliberately not
matching 9 bare digits with no separators, since that shape collides
with too many other identifiers (invoice numbers, zip+4, etc.) to be a
useful signal on its own."""

_CREDIT_CARD_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn_checksum(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


class RegexPiiDetector:
    """Default PiiDetector. See module docstring for each pattern's
    known false-positive/false-negative shape."""

    def detect(self, text: str) -> list[PiiSpan]:
        spans: list[PiiSpan] = []
        for m in _EMAIL_RE.finditer(text):
            spans.append(PiiSpan(pii_type="email", start=m.start(), end=m.end()))
        for m in _PHONE_RE.finditer(text):
            spans.append(PiiSpan(pii_type="phone", start=m.start(), end=m.end()))
        for m in _SSN_RE.finditer(text):
            spans.append(PiiSpan(pii_type="ssn", start=m.start(), end=m.end()))
        for m in _CREDIT_CARD_CANDIDATE_RE.finditer(text):
            digits = re.sub(r"[ -]", "", m.group())
            if 13 <= len(digits) <= 19 and _luhn_checksum(digits):
                spans.append(PiiSpan(pii_type="credit_card", start=m.start(), end=m.end()))
        return spans
