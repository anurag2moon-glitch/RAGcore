"""Format-specific Loaders and the registry that dispatches on file
extension.

Each format loader imports its native library lazily (inside the loader
module) and raises LoaderError with an install hint if the optional
dependency is missing — so `import ragcore` never requires pymupdf,
ebooklib, or python-docx unless that format is actually used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ragcore.errors import LoaderError
from ragcore.loaders.base import Loader
from ragcore.loaders.docx import DocxLoader
from ragcore.loaders.epub import EpubLoader
from ragcore.loaders.html import HtmlLoader
from ragcore.loaders.pdf import PdfLoader
from ragcore.loaders.text import MarkdownLoader, TextLoader

_DEFAULT_LOADERS: tuple[Loader, ...] = (
    TextLoader(),
    MarkdownLoader(),
    PdfLoader(),
    EpubLoader(),
    DocxLoader(),
    HtmlLoader(),
)


class LoaderRegistry:
    """Dispatches a file path to the Loader registered for its extension."""

    def __init__(self, loaders: Iterable[Loader] | None = None) -> None:
        self._by_ext: dict[str, Loader] = {}
        for loader in loaders if loaders is not None else _DEFAULT_LOADERS:
            for ext in loader.extensions:
                self._by_ext[ext] = loader

    def register(self, loader: Loader) -> None:
        for ext in loader.extensions:
            self._by_ext[ext] = loader

    def get_loader(self, path: Path) -> Loader:
        ext = path.suffix.lower()
        loader = self._by_ext.get(ext)
        if loader is None:
            raise LoaderError(
                f"no loader registered for extension {ext!r} ({path}); "
                f"supported: {sorted(self._by_ext)}"
            )
        return loader


__all__ = [
    "Loader",
    "LoaderRegistry",
    "TextLoader",
    "MarkdownLoader",
    "PdfLoader",
    "EpubLoader",
    "DocxLoader",
    "HtmlLoader",
]
