"""PDF loader, backed by PyMuPDF (fitz).

Structure comes from the PDF's outline/table of contents (`get_toc()`),
which gives (level, title, page) entries. We walk pages in order,
maintaining a breadcrumb stack updated whenever a TOC entry's page is
reached, and tag each page's Block with the active breadcrumb and page
number. A PDF with no embedded outline degrades to page-only structure
(section_path == ()) — still useful, since page_start/page_end survive.

A page with no embedded text layer (a scanned page: PyMuPDF's
`get_text("text")` returns empty) falls back to OCR when `ocr=True` (the
default): the page is rendered to a PNG pixmap and handed to `ocr_fn`
(default `_tesseract_ocr`, lazily imported so `ragcore[pdf]` alone never
requires pytesseract/Pillow/the tesseract binary for an all-text-layer
corpus). `ocr_fn` is injectable so tests and alternative OCR backends
don't need the real tesseract binary installed.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Callable

from ragcore.errors import LoaderError
from ragcore.types import Block, Document

try:
    import pymupdf as fitz  # PyMuPDF's current import name; `fitz` is deprecated
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    fitz = None
else:
    _IMPORT_ERROR = None

OcrFn = Callable[[bytes, str], str]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tesseract_ocr(image_bytes: bytes, lang: str) -> str:
    """Default `ocr_fn`: render bytes (PNG) -> text via pytesseract/Tesseract."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:
        raise LoaderError(
            "OCR fallback requires the 'ocr' extra: install ragcore[ocr]"
        ) from e

    try:
        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(image, lang=lang)
    except pytesseract.TesseractNotFoundError as e:
        raise LoaderError(
            "OCR fallback requires the tesseract binary on PATH (the 'ocr' "
            "extra installs only the Python wrapper, not the OCR engine "
            "itself — see https://tesseract-ocr.github.io/tessdoc/Installation.html)"
        ) from e


class PdfLoader:
    extensions = frozenset({".pdf"})

    def __init__(
        self,
        ocr: bool = True,
        ocr_lang: str = "eng",
        ocr_dpi: int = 300,
        ocr_fn: OcrFn | None = None,
    ) -> None:
        self._ocr = ocr
        self._ocr_lang = ocr_lang
        self._ocr_dpi = ocr_dpi
        self._ocr_fn = ocr_fn or _tesseract_ocr

    def load(self, path: Path) -> Document:
        if fitz is None:
            raise LoaderError(
                "PDF loading requires the 'pdf' extra: install ragcore[pdf]"
            ) from _IMPORT_ERROR

        try:
            doc = fitz.open(path)
        except Exception as e:  # noqa: BLE001 - re-raised as a typed error
            raise LoaderError(f"could not open PDF {path}: {e}") from e

        try:
            toc = doc.get_toc(simple=True)  # list of [level, title, page(1-indexed)]
        except Exception:  # noqa: BLE001 - a broken outline shouldn't fail the load
            toc = []

        # entries_by_page[page] = list of (level, title), in TOC order
        entries_by_page: dict[int, list[tuple[int, str]]] = {}
        for level, title, page in toc:
            entries_by_page.setdefault(page, []).append((level, title))

        title = (doc.metadata or {}).get("title") or path.stem
        blocks: list[Block] = []
        stack: list[str] = []

        for page_index in range(doc.page_count):
            page_num = page_index + 1  # 1-indexed, matches TOC convention
            for level, heading in entries_by_page.get(page_num, []):
                del stack[level - 1 :]
                while len(stack) < level - 1:
                    stack.append("")
                stack.append(heading)

            page_obj = doc[page_index]
            text = page_obj.get_text("text").strip()
            if not text and self._ocr:
                pixmap = page_obj.get_pixmap(dpi=self._ocr_dpi)
                text = self._ocr_fn(pixmap.tobytes("png"), self._ocr_lang).strip()
            if text:
                blocks.append(
                    Block(
                        text=text,
                        page=page_num,
                        section_path=tuple(stack),
                        page_hash=_sha256_text(text),
                    )
                )

        meta = {k: v for k, v in (doc.metadata or {}).items() if v}
        doc.close()

        return Document(
            source_id=_sha256_file(path),
            path=str(path),
            title=title,
            source_type="pdf",
            blocks=blocks,
            meta=meta,
        )
