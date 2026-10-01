"""Shared fixtures: build small structured "book" files (one per format)
with parallel content, so loader tests can assert the same structural facts
(chapter/section breadcrumb, page numbers) across formats.

Book shape used everywhere below:
    Chapter 1: Introduction
        (body paragraph)
        Background          <- subsection
            (body paragraph)
    Chapter 2: Methods
        (body paragraph)
"""

from __future__ import annotations

from pathlib import Path

import pytest

CH1_TITLE = "Chapter 1: Introduction"
CH1_BODY = "This chapter introduces contextualized retrieval over long documents."
SEC_TITLE = "Background"
SEC_BODY = "Background covers prior work on chunking and embeddings."
CH2_TITLE = "Chapter 2: Methods"
CH2_BODY = "This chapter describes the structural chunking method in detail."


@pytest.fixture
def md_book(tmp_path: Path) -> Path:
    path = tmp_path / "book.md"
    path.write_text(
        f"# {CH1_TITLE}\n\n{CH1_BODY}\n\n## {SEC_TITLE}\n\n{SEC_BODY}\n\n# {CH2_TITLE}\n\n{CH2_BODY}\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def txt_book(tmp_path: Path) -> Path:
    path = tmp_path / "plain.txt"
    path.write_text("Plain text with no headings, used to test the no-structure case.", encoding="utf-8")
    return path


@pytest.fixture
def docx_book(tmp_path: Path) -> Path:
    docx = pytest.importorskip("docx")
    path = tmp_path / "book.docx"
    doc = docx.Document()
    doc.add_heading(CH1_TITLE, level=1)
    doc.add_paragraph(CH1_BODY)
    doc.add_heading(SEC_TITLE, level=2)
    doc.add_paragraph(SEC_BODY)
    doc.add_heading(CH2_TITLE, level=1)
    doc.add_paragraph(CH2_BODY)
    doc.save(str(path))
    return path


@pytest.fixture
def pdf_book(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("pymupdf")
    path = tmp_path / "book.pdf"
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((72, 72), CH1_TITLE, fontsize=16)
    page1.insert_text((72, 110), CH1_BODY, fontsize=11)
    page1.insert_text((72, 140), SEC_TITLE, fontsize=13)
    page1.insert_text((72, 170), SEC_BODY, fontsize=11)

    page2 = doc.new_page()
    page2.insert_text((72, 72), CH2_TITLE, fontsize=16)
    page2.insert_text((72, 110), CH2_BODY, fontsize=11)

    # page numbers in set_toc are 1-indexed
    doc.set_toc(
        [
            [1, CH1_TITLE, 1],
            [2, SEC_TITLE, 1],
            [1, CH2_TITLE, 2],
        ]
    )
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def scanned_pdf_book(tmp_path: Path) -> Path:
    """A one-page PDF with an image and no embedded text layer at all —
    get_text("text") returns "", the condition PdfLoader's OCR fallback
    triggers on."""
    fitz = pytest.importorskip("pymupdf")
    path = tmp_path / "scanned.pdf"
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100))
    pix.set_rect(pix.irect, (255, 255, 255))
    page.insert_image(fitz.Rect(72, 72, 172, 172), pixmap=pix)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def epub_book(tmp_path: Path) -> Path:
    ebooklib = pytest.importorskip("ebooklib")
    from ebooklib import epub

    path = tmp_path / "book.epub"
    book = epub.EpubBook()
    book.set_identifier("ragcore-test-book")
    book.set_title("Sample Book")
    book.set_language("en")

    c1 = epub.EpubHtml(title=CH1_TITLE, file_name="ch1.xhtml", lang="en")
    c1.content = f"<h1>{CH1_TITLE}</h1><p>{CH1_BODY}</p><h2>{SEC_TITLE}</h2><p>{SEC_BODY}</p>"
    book.add_item(c1)

    c2 = epub.EpubHtml(title=CH2_TITLE, file_name="ch2.xhtml", lang="en")
    c2.content = f"<h1>{CH2_TITLE}</h1><p>{CH2_BODY}</p>"
    book.add_item(c2)

    book.toc = (
        epub.Link("ch1.xhtml", CH1_TITLE, "ch1"),
        epub.Link("ch2.xhtml", CH2_TITLE, "ch2"),
    )
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", c1, c2]

    epub.write_epub(str(path), book)
    return path


@pytest.fixture
def html_book(tmp_path: Path) -> Path:
    path = tmp_path / "book.html"
    path.write_text(
        "<html><head><title>Sample Book</title>"
        "<style>.x{color:red}</style></head><body>"
        f"<h1>{CH1_TITLE}</h1><p>{CH1_BODY}</p>"
        f"<h2>{SEC_TITLE}</h2><p>{SEC_BODY}</p>"
        f"<script>var x=1;</script>"
        f"<h1>{CH2_TITLE}</h1><p>{CH2_BODY}</p>"
        "</body></html>",
        encoding="utf-8",
    )
    return path
