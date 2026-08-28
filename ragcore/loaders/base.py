"""Loader protocol: file bytes on disk -> a structured Document."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from ragcore.types import Document


@runtime_checkable
class Loader(Protocol):
    """Extracts a Document (flat, ordered Blocks with structure breadcrumbs)
    from one file on disk.

    Implementations are format-specific (pdf.py, epub.py, docx.py, text.py)
    and are looked up by extension via LoaderRegistry. A Loader does not
    chunk or embed — it only extracts text and structure.
    """

    extensions: ClassVar[frozenset[str]]
    """File extensions this loader handles, including the leading dot,
    e.g. frozenset({".pdf"})."""

    def load(self, path: Path) -> Document: ...
