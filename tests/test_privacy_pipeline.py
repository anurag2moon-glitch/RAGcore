"""scan_corpus: runs a PiiDetector over an ingested Corpus's chunks and
anchors each PiiSpan back to its chunk_id/source_id; source_ids scoping
(the mechanism a TenantScopedCorpus-aware caller would use to scan only
its own tenant's sources).
"""

from __future__ import annotations

from pathlib import Path

from ragcore.corpus import Corpus
from ragcore.privacy.pipeline import scan_corpus
from ragcore.privacy.regex_detector import RegexPiiDetector
from tests.fakes import FakeEmbedder, InMemoryStore


def test_scan_corpus_counts_every_chunk(md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)

    report = scan_corpus(corpus, RegexPiiDetector())

    assert report.chunks_scanned == corpus.count()
    assert report.chunks_scanned > 0


def test_scan_corpus_finds_a_real_email_in_ingested_text(tmp_path: Path) -> None:
    p = tmp_path / "book.md"
    p.write_text("# Chapter 1\n\nContact the author at jane.doe@example.com for details.\n")
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(p)

    report = scan_corpus(corpus, RegexPiiDetector())

    assert report.counts_by_type() == {"email": 1}
    assert report.matches[0].pii_type == "email"
    assert report.matches[0].source_id == corpus.sources()[0].source_id


def test_scan_corpus_match_never_carries_the_raw_matched_text(tmp_path: Path) -> None:
    """PiiMatch deliberately has no text/preview field - see its own
    docstring for why a scan report must never become a second place
    the flagged PII itself ends up persisted in plaintext."""
    p = tmp_path / "book.md"
    p.write_text("# Chapter 1\n\nContact jane.doe@example.com for details.\n")
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(p)

    report = scan_corpus(corpus, RegexPiiDetector())

    assert not hasattr(report.matches[0], "text")
    assert not hasattr(report.matches[0], "preview")


def test_scan_corpus_source_ids_restricts_the_scan(md_book: Path, tmp_path: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    book_b = tmp_path / "book_b.md"
    book_b.write_text(
        md_book.read_text(encoding="utf-8") + "\n\nExtra unique text for book b.\n",
        encoding="utf-8",
    )
    corpus.ingest(md_book)
    corpus.ingest(book_b)
    only_source = corpus.sources()[0].source_id
    expected_chunks = len(corpus.store.get_by_source(only_source))

    report = scan_corpus(corpus, RegexPiiDetector(), source_ids=[only_source])

    assert report.chunks_scanned == expected_chunks
    assert report.chunks_scanned < corpus.count()


def test_scan_corpus_report_to_dict_never_includes_a_global_text_field(md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)

    report = scan_corpus(corpus, RegexPiiDetector())
    d = report.to_dict()

    assert set(d.keys()) == {"chunks_scanned", "counts_by_type", "matches"}
    for m in d["matches"]:
        assert set(m.keys()) == {"pii_type", "chunk_id", "source_id", "start", "end"}
