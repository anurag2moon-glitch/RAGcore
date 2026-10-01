"""OCR fallback for scanned PDF pages (no embedded text layer).

Uses PdfLoader's injectable `ocr_fn` seam rather than the real tesseract
binary — see ragcore/loaders/pdf.py's module docstring for why that seam
exists. The one real-tesseract-dependent code path (`_tesseract_ocr`
itself) is exercised only for its two error-wrapping branches, which don't
need a real image or binary to trigger.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_scanned_page_falls_back_to_ocr(scanned_pdf_book: Path) -> None:
    from ragcore.loaders.pdf import PdfLoader

    calls: list[tuple[bytes, str]] = []

    def fake_ocr(image_bytes: bytes, lang: str) -> str:
        calls.append((image_bytes, lang))
        return "OCR'd page text"

    doc = PdfLoader(ocr_fn=fake_ocr).load(scanned_pdf_book)

    assert len(doc.blocks) == 1
    assert doc.blocks[0].text == "OCR'd page text"
    assert doc.blocks[0].page == 1
    assert len(calls) == 1
    image_bytes, lang = calls[0]
    assert image_bytes.startswith(b"\x89PNG")
    assert lang == "eng"


def test_scanned_page_with_ocr_disabled_yields_no_block(scanned_pdf_book: Path) -> None:
    from ragcore.loaders.pdf import PdfLoader

    doc = PdfLoader(ocr=False).load(scanned_pdf_book)
    assert doc.blocks == []


def test_text_page_never_invokes_ocr(pdf_book: Path) -> None:
    from ragcore.loaders.pdf import PdfLoader

    def fail_if_called(image_bytes: bytes, lang: str) -> str:
        raise AssertionError("ocr_fn must not be called for a page with a text layer")

    doc = PdfLoader(ocr_fn=fail_if_called).load(pdf_book)
    assert len(doc.blocks) == 2


def test_tesseract_ocr_missing_extra_raises_loader_error() -> None:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("pytesseract is installed in this environment; nothing to assert")

    from ragcore.errors import LoaderError
    from ragcore.loaders.pdf import _tesseract_ocr

    with pytest.raises(LoaderError, match=r"ragcore\[ocr\]"):
        _tesseract_ocr(b"not a real image", "eng")
