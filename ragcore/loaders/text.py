"""Loaders for plain text (.txt) and Markdown (.md).

.txt has no structure to extract: the whole file becomes a single Block
with an empty section_path, and the Chunker is responsible for splitting it
by paragraph/sentence within its token budget.

.md has ATX headings (`#`, `##`, ...), which become the section_path
breadcrumb. Each heading starts a new Block containing the text up to the
next heading of the same or shallower level.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ragcore.types import Block, Document

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _title_from_stem(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip() or path.name


class TextLoader:
    """Loads .txt files as a single unstructured Block."""

    extensions = frozenset({".txt"})

    def load(self, path: Path) -> Document:
        text = path.read_text(encoding="utf-8", errors="replace")
        return Document(
            source_id=_sha256_file(path),
            path=str(path),
            title=_title_from_stem(path),
            source_type="txt",
            blocks=[Block(text=text, page=None, section_path=())],
            meta={},
        )


class MarkdownLoader:
    """Loads .md files, splitting on ATX headings into structure-tagged
    Blocks. A leading preamble before the first heading (if any) becomes
    its own Block with an empty section_path.
    """

    extensions = frozenset({".md", ".markdown"})

    def load(self, path: Path) -> Document:
        raw = path.read_text(encoding="utf-8", errors="replace")
        lines = raw.splitlines()

        blocks: list[Block] = []
        stack: list[str] = []  # current heading breadcrumb, one entry per level
        buf: list[str] = []
        title: str | None = None

        def flush() -> None:
            text = "\n".join(buf).strip()
            if text:
                blocks.append(Block(text=text, page=None, section_path=tuple(stack)))
            buf.clear()

        for line in lines:
            m = _ATX_HEADING.match(line)
            if m:
                flush()
                level = len(m.group(1))
                heading = m.group(2)
                if title is None and level == 1:
                    title = heading
                # trim the breadcrumb to this heading's depth, then append
                del stack[level - 1 :]
                while len(stack) < level - 1:
                    stack.append("")  # unlabeled intermediate level
                stack.append(heading)
            else:
                buf.append(line)
        flush()

        if not blocks:
            blocks = [Block(text=raw.strip(), page=None, section_path=())]

        return Document(
            source_id=_sha256_file(path),
            path=str(path),
            title=title or _title_from_stem(path),
            source_type="md",
            blocks=blocks,
            meta={},
        )
