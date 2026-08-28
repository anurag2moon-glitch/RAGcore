"""Loader structure extraction, per format, against the parallel fixture
book defined in conftest.py. This is the load-bearing check from the
design doc's verification plan: structure metadata must survive extraction
before it can survive the rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import CH1_BODY, CH1_TITLE, CH2_BODY, CH2_TITLE, SEC_BODY, SEC_TITLE


def test_text_loader_has_no_structure(txt_book: Path) -> None:
    from ragcore.loaders.text import TextLoader

    doc = TextLoader().load(txt_book)
    assert doc.source_type == "txt"
    assert len(doc.blocks) == 1
    assert doc.blocks[0].section_path == ()
    assert doc.blocks[0].page is None
    assert "no headings" in doc.blocks[0].text


def test_markdown_loader_extracts_heading_breadcrumbs(md_book: Path) -> None:
    from ragcore.loaders.text import MarkdownLoader

    doc = MarkdownLoader().load(md_book)
    assert doc.source_type == "md"
    assert doc.title == CH1_TITLE

    by_text = {b.section_path: b.text for b in doc.blocks}
    assert by_text[(CH1_TITLE,)] == CH1_BODY
    assert by_text[(CH1_TITLE, SEC_TITLE)] == SEC_BODY
    assert by_text[(CH2_TITLE,)] == CH2_BODY


def test_docx_loader_extracts_heading_breadcrumbs(docx_book: Path) -> None:
    from ragcore.loaders.docx import DocxLoader

    doc = DocxLoader().load(docx_book)
    assert doc.source_type == "docx"
    assert doc.title == CH1_TITLE

    by_text = {b.section_path: b.text for b in doc.blocks}
    assert by_text[(CH1_TITLE,)] == CH1_BODY
    assert by_text[(CH1_TITLE, SEC_TITLE)] == SEC_BODY
    assert by_text[(CH2_TITLE,)] == CH2_BODY


def test_pdf_loader_extracts_outline_and_pages(pdf_book: Path) -> None:
    from ragcore.loaders.pdf import PdfLoader

    doc = PdfLoader().load(pdf_book)
    assert doc.source_type == "pdf"
    assert len(doc.blocks) == 2

    page1, page2 = doc.blocks
    assert page1.page == 1
    assert page1.section_path == (CH1_TITLE, SEC_TITLE)
    assert CH1_BODY in page1.text
    assert SEC_BODY in page1.text

    assert page2.page == 2
    assert page2.section_path == (CH2_TITLE,)
    assert CH2_BODY in page2.text


def test_epub_loader_extracts_toc_breadcrumbs(epub_book: Path) -> None:
    from ragcore.loaders.epub import EpubLoader

    doc = EpubLoader().load(epub_book)
    assert doc.source_type == "epub"

    ch1_blocks = [b for b in doc.blocks if CH1_BODY in b.text]
    ch2_blocks = [b for b in doc.blocks if CH2_BODY in b.text]
    assert len(ch1_blocks) == 1
    assert len(ch2_blocks) == 1
    assert ch1_blocks[0].section_path[0] == CH1_TITLE
    assert ch2_blocks[0].section_path[0] == CH2_TITLE
    assert SEC_BODY in ch1_blocks[0].text
