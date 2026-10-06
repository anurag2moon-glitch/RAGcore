"""EPUB loader, backed by ebooklib (container/spine/TOC parsing) and
selectolax (fast HTML-to-text).

Uses `selectolax.lexbor.LexborHTMLParser`, not the `selectolax.parser`
top-level import: selectolax 1.0 deprecated and disabled that path's
"Modest" backend outright (`ImportError` at import time, not a
DeprecationWarning), discovered by actually installing this package into
a clean environment while setting up CI (docs/ARCHITECTURE.md Phase 6
item 23) rather than trusting an already-populated dev venv pinned to an
older selectolax. `LexborHTMLParser` has an identical API for everything
this loader uses (`.body`, `.strip_tags()`, `.css_first()`,
`.text(separator=, strip=)`, `.html`) — confirmed directly, not assumed.

Structure comes from the EPUB's nav/TOC (`book.toc`), which maps hrefs
(optionally with a `#fragment` anchor) to chapter/section titles in a
nested tree. Many real-world EPUBs — notably Project Gutenberg's output —
pack several chapters into one XHTML file, with separate TOC entries
pointing at `#fragment` anchors inside it rather than at separate files.
So structure is resolved at two levels: files map to the TOC entries whose
href points at them (in TOC order), and each file's raw HTML is then split
at the byte offset of each such anchor's `id="..."` attribute, giving one
Block per TOC entry rather than one Block per file. Content preceding a
file's first anchor is a continuation of whatever chapter was open at the
end of the previous file (front matter before any heading gets no
breadcrumb). A file with no TOC entries at all falls back to whichever
breadcrumb was open when it started.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ragcore.errors import LoaderError
from ragcore.types import Block, Document

try:
    from ebooklib import ITEM_DOCUMENT, epub
    from selectolax.lexbor import LexborHTMLParser as HTMLParser
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


def _split_href(href: str) -> tuple[str, str]:
    """Split an href into (file_part, fragment_part). fragment_part is ''
    when the href has no `#...` suffix."""
    file_part, _, fragment = href.partition("#")
    return file_part, fragment


def _flatten_toc(
    toc, breadcrumb: tuple[str, ...], out: dict[str, list[tuple[str, tuple[str, ...]]]]
) -> None:
    """Recursively flatten ebooklib's nested TOC into
    file_href -> [(fragment_id, breadcrumb), ...] in TOC document order.
    A TOC entry whose href has no fragment is recorded with fragment_id ''
    (meaning: applies from the start of that file).
    """
    for entry in toc:
        if isinstance(entry, tuple):
            node, children = entry  # (Section/Link, [children])
        else:
            node, children = entry, ()
        title = getattr(node, "title", None) or ""
        href = getattr(node, "href", None)
        crumb = breadcrumb + (title,) if title else breadcrumb
        if href:
            file_part, fragment = _split_href(href)
            out.setdefault(file_part, []).append((fragment, crumb))
        _flatten_toc(children, crumb, out)


def _find_anchor_offset(html: str, anchor_id: str) -> int | None:
    """Byte offset of the *start* of the tag carrying `id="anchor_id"` (or
    `id='...'`) in the raw HTML source — not the offset of the `id=`
    attribute itself. Slicing at the attribute's offset would cut through
    the middle of an opening tag (e.g. `<h2 id="x">`), leaving a dangling
    `id="x">` that a lenient HTML parser treats as literal text rather than
    markup, leaking it into the extracted chunk text. Regex over the raw
    source rather than a DOM lookup, since we need a linear position to
    slice on, not a node.
    """
    m = re.search(r'id=["\']' + re.escape(anchor_id) + r'["\']', html)
    if m is None:
        return None
    tag_start = html.rfind("<", 0, m.start())
    return tag_start if tag_start != -1 else m.start()


def _html_to_text(html_fragment: str) -> str:
    parser = HTMLParser(html_fragment)
    body = parser.body or parser.html
    if body is None:
        return ""
    return body.text(separator="\n", strip=True).strip()


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

        toc_by_file: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
        _flatten_toc(book.toc, (), toc_by_file)

        title = None
        dc_title = book.get_metadata("DC", "title")
        if dc_title:
            title = dc_title[0][0]

        blocks: list[Block] = []
        current_crumb: tuple[str, ...] = ()

        for item in book.get_items_of_type(ITEM_DOCUMENT):
            file_href, _ = _split_href(item.get_name())
            raw_html = item.get_content().decode("utf-8", errors="replace")
            entries = toc_by_file.get(file_href, [])

            # Resolve each TOC entry for this file to a byte offset; entries
            # whose anchor isn't found (shouldn't normally happen) are dropped.
            located: list[tuple[int, tuple[str, ...]]] = []
            for fragment, crumb in entries:
                if not fragment:
                    located.append((0, crumb))
                    continue
                offset = _find_anchor_offset(raw_html, fragment)
                if offset is not None:
                    located.append((offset, crumb))
            located.sort(key=lambda pair: pair[0])

            if not located:
                # No TOC entry touches this file: it's a continuation of
                # whatever chapter was open when the previous file ended.
                text = _html_to_text(raw_html)
                if text:
                    blocks.append(Block(text=text, page=None, section_path=current_crumb))
                continue

            if located[0][0] > 0:
                leading_text = _html_to_text(raw_html[: located[0][0]])
                if leading_text:
                    blocks.append(Block(text=leading_text, page=None, section_path=current_crumb))

            for i, (offset, crumb) in enumerate(located):
                end = located[i + 1][0] if i + 1 < len(located) else len(raw_html)
                segment_text = _html_to_text(raw_html[offset:end])
                if segment_text:
                    blocks.append(Block(text=segment_text, page=None, section_path=crumb))

            current_crumb = located[-1][1]

        return Document(
            source_id=_sha256_file(path),
            path=str(path),
            title=title,
            source_type="epub",
            blocks=blocks,
            meta={},
        )
