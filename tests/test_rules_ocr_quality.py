"""is_low_quality_ocr: the two real OCR failure signatures it catches
(replacement characters, low alphabetic ratio) and the short-text
exemption — Phase 10 §2.
"""

from __future__ import annotations

from ragcore.rules.ocr_quality import is_low_quality_ocr


def test_clean_text_is_not_flagged() -> None:
    text = "Mars in the eighth house causes health troubles during Mars dasha."
    assert is_low_quality_ocr(text) is False


def test_text_with_replacement_characters_is_flagged() -> None:
    text = "Mar� in ��� eighth h�use c��ses h�alth �roubles today"
    assert is_low_quality_ocr(text) is True


def test_text_with_a_low_alphabetic_ratio_is_flagged() -> None:
    text = "!@#$%^&*()_+-=][;/.,<>?~`|}{\":" * 3
    assert is_low_quality_ocr(text) is True


def test_short_text_is_never_flagged_regardless_of_content() -> None:
    assert is_low_quality_ocr("!@#$%") is False
    assert is_low_quality_ocr("") is False


def test_whitespace_only_text_is_not_flagged() -> None:
    assert is_low_quality_ocr(" " * 30) is False


def test_custom_thresholds_are_respected() -> None:
    borderline = "ab" + "1" * 28  # 2/30 alphabetic = 0.067, above min_length
    assert is_low_quality_ocr(borderline, min_alpha_ratio=0.05) is False
    assert is_low_quality_ocr(borderline, min_alpha_ratio=0.5) is True
