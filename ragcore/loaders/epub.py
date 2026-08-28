"""EPUB loader, backed by ebooklib (container/spine/TOC parsing) and
selectolax (fast HTML-to-text).

Structure comes from the EPUB's nav/TOC (`book.toc`), which maps hrefs to
chapter/section titles in a nested tree. We flatten that tree into
href -> breadcrumb, then walk the spine (the book's linear reading order)
and tag each spine document's Block with the breadcrumb of the closest TOC
entry pointing at it (fragment identifiers are ignored — TOC entries are
matched at file granularity, which is what EPUB chapter splits use in
practice).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ragcore.errors import LoaderError
from ragcore.types import Block, Document

try:
    from ebooklib import ITEM_DOCUMENT, epub
    from selectolax.parser import HTMLParser
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    epub = None
    HTMLParser = None
else:
    _IMPORT_ERROR = None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_fragment(href: str) -> str:
    return href.split("#", 1)[0]


def _flatten_toc(toc, breadcrumb: tuple[str, ...], out: dict[str, tuple[str, ...]]) -> None:
    """Recursively flatten ebooklib's nested TOC into href -> breadcrumb."""
    for entry in toc:
        if isinstance(entry, tuple):
            # (Section/Link, [children])
            node, children = entry
            title = getattr(node, "title", None) or ""
            href = getattr(node, "href", None)
            crumb = breadcrumb + (title,) if title else breadcrumb
            if href:
                out.setdefault(_strip_fragment(href), crumb)
            _flatten_toc(children, crumb, out)
        else:
            title = getattr(entry, "title", None) or ""
            href = getattr(entry, "href", None)
            crumb = breadcrumb + (title,) if title else breadcrumb
            if href:
                out.setdefault(_strip_fragment(href), crumb)


class EpubLoader:
    extensions = frozenset({".epub"})

    def load(self, path: Path) -> Document:
        if epub is None:
            raise LoaderError(
                "EPUB loading requires the 'epub' extra: install ragcore[epub]"
            ) from _IMPORT_ERROR

        try:
            book = epub.read_epub(str(path))
        except Exception as e:  # noqa: BLE001 - re-raised as a typed error
            raise LoaderError(f"could not open EPUB {path}: {e}") from e

        href_to_crumb: dict[str, tuple[str, ...]] = {}
        _flatten_toc(book.toc, (), href_to_crumb)

        title = None
        dc_title = book.get_metadata("DC", "title")
        if dc_title:
            title = dc_title[0][0]

        blocks: list[Block] = []
        current_crumb: tuple[str, ...] = ()

        for item in book.get_items_of_type(ITEM_DOCUMENT):
            href = _strip_fragment(item.get_name())
            current_crumb = href_to_crumb.get(href, current_crumb)

            html = HTMLParser(item.get_content())
            text = html.body.text(separator="\n", strip=True) if html.body else ""
            text = text.strip()
            if text:
                blocks.append(Block(text=text, page=None, section_path=current_crumb))

        return Document(
            source_id=_sha256_file(path),
            path=str(path),
            title=title,
            source_type="epub",
            blocks=blocks,
            meta={},
        )
