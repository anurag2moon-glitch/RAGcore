"""Loader for standalone HTML files (.html, .htm), backed by selectolax.

Splits on <h1>-<h6> headings into structure-tagged Blocks, using the same
breadcrumb-stack semantics as MarkdownLoader/DocxLoader (a heading opens a
new level; body text accumulates under the current breadcrumb until the
next heading). Unlike those two, it locates headings positionally over the
raw markup — the same slice-and-convert technique EpubLoader already uses
for Project Gutenberg's XHTML — rather than walking the DOM node-by-node,
since a naive DOM walk that also descends into nested block elements (e.g.
`<blockquote><p>...</p></blockquote>`) risks visiting both and duplicating
text. <script>/<style> contents are stripped before the heading regex runs,
so inline script text containing a literal "<h1>" string can't be
mistaken for structure.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ragcore.errors import LoaderError
from ragcore.types import Block, Document

try:
    from selectolax.parser import HTMLParser
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    HTMLParser = None
else:
    _IMPORT_ERROR = None

_HEADING = re.compile(r"<h([1-6])\b[^>]*>(.*?)</h\1\s*>", re.IGNORECASE | re.DOTALL)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _html_to_text(html_fragment: str) -> str:
    parser = HTMLParser(html_fragment)
    body = parser.body or parser.html
    if body is None:
        return ""
    return body.text(separator="\n", strip=True).strip()


def _title_from_stem(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip() or path.name


class HtmlLoader:
    """Loads a standalone .html/.htm file as heading-delimited Blocks."""

    extensions = frozenset({".html", ".htm"})

    def load(self, path: Path) -> Document:
        if HTMLParser is None:
            raise LoaderError(
                "HTML loading requires the 'html' extra: install ragcore[html]"
            ) from _IMPORT_ERROR

        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            parser = HTMLParser(raw)
        except Exception as e:  # noqa: BLE001 - re-raised as a typed error
            raise LoaderError(f"could not parse HTML {path}: {e}") from e

        parser.strip_tags(["script", "style"])
        title_node = parser.css_first("title")
        title = title_node.text(strip=True) if title_node else None

        body_node = parser.body
        clean_html = (body_node.html if body_node is not None else parser.html) or ""

        headings = list(_HEADING.finditer(clean_html))

        blocks: list[Block] = []
        stack: list[str] = []

        if not headings:
            text = _html_to_text(clean_html)
            if text:
                blocks.append(Block(text=text, page=None, section_path=()))
            return Document(
                source_id=_sha256_file(path),
                path=str(path),
                title=title or _title_from_stem(path),
                source_type="html",
                blocks=blocks,
                meta={},
            )

        if headings[0].start() > 0:
            leading_text = _html_to_text(clean_html[: headings[0].start()])
            if leading_text:
                blocks.append(Block(text=leading_text, page=None, section_path=()))

        for i, m in enumerate(headings):
            level = int(m.group(1))
            heading_text = _html_to_text(m.group(2))

            del stack[level - 1 :]
            while len(stack) < level - 1:
                stack.append("")
            stack.append(heading_text)

            body_start = m.end()
            body_end = headings[i + 1].start() if i + 1 < len(headings) else len(clean_html)
            section_text = _html_to_text(clean_html[body_start:body_end])
            if section_text:
                blocks.append(Block(text=section_text, page=None, section_path=tuple(stack)))

        return Document(
            source_id=_sha256_file(path),
            path=str(path),
            title=title or _title_from_stem(path),
            source_type="html",
            blocks=blocks,
            meta={},
        )
