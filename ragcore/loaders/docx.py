"""DOCX loader, backed by python-docx.

Structure comes from paragraph styles named "Heading N" (Word's standard
heading styles). Each heading starts a new Block; body paragraphs
accumulate under the current breadcrumb until the next heading.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ragcore.errors import LoaderError
from ragcore.types import Block, Document

try:
    import docx as python_docx
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    python_docx = None
else:
    _IMPORT_ERROR = None

_HEADING_STYLE = re.compile(r"^Heading (\d+)$", re.IGNORECASE)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class DocxLoader:
    extensions = frozenset({".docx"})

    def load(self, path: Path) -> Document:
        if python_docx is None:
            raise LoaderError(
                "DOCX loading requires the 'docx' extra: install ragcore[docx]"
            ) from _IMPORT_ERROR

        try:
            doc = python_docx.Document(str(path))
        except Exception as e:  # noqa: BLE001 - re-raised as a typed error
            raise LoaderError(f"could not open DOCX {path}: {e}") from e

        blocks: list[Block] = []
        stack: list[str] = []
        buf: list[str] = []
        title: str | None = None

        def flush() -> None:
            text = "\n".join(buf).strip()
            if text:
                blocks.append(Block(text=text, page=None, section_path=tuple(stack)))
            buf.clear()

        for para in doc.paragraphs:
            text = para.text.strip()
            m = _HEADING_STYLE.match(para.style.name or "") if para.style else None
            if m and text:
                flush()
                level = int(m.group(1))
                if title is None and level == 1:
                    title = text
                del stack[level - 1 :]
                while len(stack) < level - 1:
                    stack.append("")
                stack.append(text)
            elif text:
                buf.append(text)
        flush()

        core_props = doc.core_properties
        meta = {
            k: v
            for k, v in {
                "author": core_props.author,
                "subject": core_props.subject,
            }.items()
            if v
        }

        return Document(
            source_id=_sha256_file(path),
            path=str(path),
            title=title or core_props.title or path.stem,
            source_type="docx",
            blocks=blocks,
            meta=meta,
        )
