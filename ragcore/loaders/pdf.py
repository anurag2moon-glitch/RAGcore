"""PDF loader, backed by PyMuPDF (fitz).

Structure comes from the PDF's outline/table of contents (`get_toc()`),
which gives (level, title, page) entries. We walk pages in order,
maintaining a breadcrumb stack updated whenever a TOC entry's page is
reached, and tag each page's Block with the active breadcrumb and page
number. A PDF with no embedded outline degrades to page-only structure
(section_path == ()) — still useful, since page_start/page_end survive.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ragcore.errors import LoaderError
from ragcore.types import Block, Document

try:
    import pymupdf as fitz  # PyMuPDF's current import name; `fitz` is deprecated
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    fitz = None
else:
    _IMPORT_ERROR = None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class PdfLoader:
    extensions = frozenset({".pdf"})

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

            text = doc[page_index].get_text("text").strip()
            if text:
                blocks.append(
                    Block(text=text, page=page_num, section_path=tuple(stack))
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
