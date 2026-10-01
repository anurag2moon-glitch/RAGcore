"""Page-level content-hash dedupe: Block.page_hash (set by PdfLoader, the
only loader with a real page concept) and StructuralChunker's propagation
of it into ChunkMeta.extra["page_hash"] for a chunk that doesn't straddle
a page boundary.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ragcore.chunking.structural import StructuralChunker
from ragcore.loaders.pdf import PdfLoader


def test_pdf_loader_sets_page_hash_per_page(pdf_book: Path) -> None:
    doc = PdfLoader().load(pdf_book)
    assert len(doc.blocks) == 2

    page1, page2 = doc.blocks
    assert page1.page_hash == hashlib.sha256(page1.text.encode("utf-8")).hexdigest()
    assert page2.page_hash == hashlib.sha256(page2.text.encode("utf-8")).hexdigest()
    assert page1.page_hash != page2.page_hash


def test_page_hash_is_content_addressed_not_positional(tmp_path: Path) -> None:
    """Two different PDFs whose first page has byte-identical text get the
    same page_hash — it's a hash of content, like source_id, not a
    position-derived id like chunk_id."""
    fitz = pytest.importorskip("pymupdf")
    shared_text = "This exact page of boilerplate recurs across two different PDFs."

    paths = []
    for name in ("a.pdf", "b.pdf"):
        path = tmp_path / name
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), shared_text, fontsize=11)
        doc.save(str(path))
        doc.close()
        paths.append(path)

    doc_a = PdfLoader().load(paths[0])
    doc_b = PdfLoader().load(paths[1])

    assert doc_a.source_id != doc_b.source_id  # different files
    assert doc_a.blocks[0].page_hash == doc_b.blocks[0].page_hash  # same page content


def test_non_pdf_loaders_leave_page_hash_none(md_book: Path) -> None:
    from ragcore.loaders.text import MarkdownLoader

    doc = MarkdownLoader().load(md_book)
    assert all(b.page_hash is None for b in doc.blocks)


def test_chunk_within_one_page_gets_page_hash_in_extra(pdf_book: Path) -> None:
    doc = PdfLoader().load(pdf_book)
    groups = StructuralChunker().chunk(doc)
    chunks = [c for g in groups for c in g.chunks]

    page1_hash = doc.blocks[0].page_hash
    page2_hash = doc.blocks[1].page_hash

    ch1_chunks = [c for c in chunks if c.meta.page_start == 1 and c.meta.page_end == 1]
    ch2_chunks = [c for c in chunks if c.meta.page_start == 2 and c.meta.page_end == 2]
    assert ch1_chunks and all(c.meta.extra.get("page_hash") == page1_hash for c in ch1_chunks)
    assert ch2_chunks and all(c.meta.extra.get("page_hash") == page2_hash for c in ch2_chunks)


def test_chunk_spanning_two_pages_gets_no_page_hash(tmp_path: Path) -> None:
    """Two short pages in the same chapter (no new heading on page 2) pack
    into a single chunk under the default token budget — that chunk must
    not carry either page's hash as if it represented the whole chunk."""
    fitz = pytest.importorskip("pymupdf")
    path = tmp_path / "two_page_chapter.pdf"
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((72, 72), "Chapter One", fontsize=16)
    page1.insert_text((72, 110), "Short first page body.", fontsize=11)

    page2 = doc.new_page()
    page2.insert_text((72, 72), "Short second page body, same chapter.", fontsize=11)

    doc.set_toc([[1, "Chapter One", 1]])
    doc.save(str(path))
    doc.close()

    loaded = PdfLoader().load(path)
    assert loaded.blocks[0].section_path == loaded.blocks[1].section_path  # same chapter

    groups = StructuralChunker().chunk(loaded)
    chunks = [c for g in groups for c in g.chunks]
    straddling = [c for c in chunks if c.meta.page_start == 1 and c.meta.page_end == 2]
    assert straddling, "expected the two short pages to pack into one chunk"
    assert "page_hash" not in straddling[0].meta.extra
