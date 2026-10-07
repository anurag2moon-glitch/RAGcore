"""A text-based OCR quality heuristic — Phase 10 §2's "Flag low-quality
OCR pages instead of extracting from them."

This is deliberately a heuristic over the *text itself*, not a
confidence score from the OCR engine: `ragcore.loaders`' OCR path
(`pytesseract`, the `ocr` extra) returns plain recognized text with no
per-word/per-page confidence carried through to `Chunk`/`ChunkMeta` —
piping that through would be a `ragcore.loaders` change, out of scope
here. Two real OCR failure signatures this catches: the Unicode
replacement character (`U+FFFD`), which `pytesseract`/the underlying
image decoder emits for a glyph it couldn't map to any character at
all; and a low ratio of alphabetic characters to non-whitespace
characters, which catches garbled runs of punctuation-like noise OCR
engines produce on damaged/low-resolution scans, without the
replacement character necessarily appearing.
"""

from __future__ import annotations

REPLACEMENT_CHAR = "�"

DEFAULT_MIN_LENGTH = 20
"""Below this, there isn't enough text for the alpha-ratio signal to
be meaningful — a 3-character chunk being "0% alphabetic" by chance
isn't evidence of OCR noise."""

DEFAULT_MAX_REPLACEMENT_RATIO = 0.01
DEFAULT_MIN_ALPHA_RATIO = 0.4


def is_low_quality_ocr(
    text: str,
    *,
    min_length: int = DEFAULT_MIN_LENGTH,
    max_replacement_ratio: float = DEFAULT_MAX_REPLACEMENT_RATIO,
    min_alpha_ratio: float = DEFAULT_MIN_ALPHA_RATIO,
) -> bool:
    if len(text) < min_length:
        return False

    replacement_ratio = text.count(REPLACEMENT_CHAR) / len(text)
    if replacement_ratio > max_replacement_ratio:
        return True

    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return False
    alpha_ratio = sum(1 for c in non_space if c.isalpha()) / len(non_space)
    return alpha_ratio < min_alpha_ratio
