"""RegexPiiDetector: email/phone/SSN/credit-card pattern detection, and the
Luhn checksum that keeps the credit-card pattern from flagging arbitrary
13-19 digit sequences.
"""

from __future__ import annotations

from ragcore.privacy.regex_detector import RegexPiiDetector


def _types(text: str) -> set[str]:
    return {s.pii_type for s in RegexPiiDetector().detect(text)}


def test_detects_an_email_address() -> None:
    spans = RegexPiiDetector().detect("reach me at jane.doe@example.com please")
    assert len(spans) == 1
    assert spans[0].pii_type == "email"


def test_detects_a_us_phone_number() -> None:
    assert "phone" in _types("call me at (555) 123-4567 tomorrow")


def test_detects_a_hyphenated_ssn() -> None:
    assert "ssn" in _types("SSN on file: 123-45-6789")


def test_does_not_flag_nine_bare_digits_as_an_ssn() -> None:
    """Deliberately conservative — see RegexPiiDetector's module docstring."""
    assert "ssn" not in _types("reference number 123456789")


def test_detects_a_luhn_valid_credit_card_number() -> None:
    # 4532015112830366 is a well-known Luhn-valid test Visa number.
    assert "credit_card" in _types("card on file: 4532 0151 1283 0366")


def test_does_not_flag_a_luhn_invalid_digit_sequence_as_a_credit_card() -> None:
    assert "credit_card" not in _types("order number 1234567890123456 was placed")


def test_spans_point_at_the_exact_matched_substring() -> None:
    text = "email me: jane.doe@example.com"
    [span] = RegexPiiDetector().detect(text)
    assert text[span.start : span.end] == "jane.doe@example.com"


def test_clean_text_with_no_pii_returns_no_spans() -> None:
    assert RegexPiiDetector().detect("the quick brown fox jumps over the lazy dog") == []


def test_multiple_pii_types_in_one_text_are_all_detected() -> None:
    text = "Contact jane.doe@example.com or (555) 123-4567, SSN 123-45-6789."
    assert _types(text) == {"email", "phone", "ssn"}
