"""build_rules: chapter grouping, the unsupported-verification-means-
never-stored rule, OCR-flagged-chunk exclusion, and resumability —
mirroring tests/test_claims.py's shape for build_claims.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Sequence

from ragcore.corpus import Corpus
from ragcore.rules.base import CandidateSpan, VerificationResult
from ragcore.rules.pipeline import build_rules
from ragcore.rules.sqlite_store import SqliteRuleStore
from ragcore.rules.types import Effect, FactPredicate, Rule, rule_content_hash
from ragcore.types import Chunk
from tests.fakes import FakeEmbedder, InMemoryStore


class _FakeOntology:
    def normalize_fact_key(self, raw_mention: str) -> str | None:
        return "fact.always"

    def normalize_fact_value(self, fact_key: str, raw_mention: str) -> Any | None:
        return 1


class _FakeDomainPack:
    domain = "test_domain"
    pack_id = "test_pack"
    pack_version = "1"
    ontology = _FakeOntology()

    def fact_types(self):
        return ()

    def extraction_system_prompt(self) -> str:
        return "extract"

    def verification_system_prompt(self) -> str:
        return "verify"


class _FakeDetector:
    model = "fake-detector"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def detect(self, chunks: Sequence[Chunk], *, domain_pack) -> list[CandidateSpan]:
        self.calls.append([c.chunk_id for c in chunks])
        chunk = chunks[0]
        return [CandidateSpan(chunk_id=chunk.chunk_id, source_id=chunk.meta.source_id, text=chunk.text)]


def _rule_for_candidate(candidate: CandidateSpan, domain: str, *, supported: bool) -> Rule:
    cond = FactPredicate(fact="fact.always", operator="eq", value=1)
    effect = Effect(category=("supported" if supported else "unsupported"), polarity="negative")
    rule_id = rule_content_hash(
        domain=domain, book_id=candidate.source_id, verbatim_span=candidate.text,
        conditions=cond, effect=effect,
    )
    return Rule(
        rule_id=rule_id, domain=domain, pack_id="test_pack", pack_version="1",
        book_id=candidate.source_id, page_start=1, chunk_ids=(candidate.chunk_id,),
        verbatim_span=candidate.text, language="en", normalized_statement="n",
        conditions=cond, effect=effect, rule_type="HARD", confidence=0.9,
        extraction_model="fake-extractor", extraction_version="v1", created_at="now",
    )


class _FakeExtractor:
    model = "fake-extractor"

    def __init__(self, *, unsupported_category: str | None = None) -> None:
        self.calls = 0
        self._unsupported_category = unsupported_category

    def extract(self, candidates: Sequence[CandidateSpan], *, domain_pack) -> list[Rule]:
        self.calls += 1
        rules = []
        for c in candidates:
            supported = self._unsupported_category != c.text
            rules.append(_rule_for_candidate(c, domain_pack.domain, supported=supported))
        return rules


class _FakeVerifier:
    model = "fake-verifier"
    _REJECT_CATEGORY: ClassVar[str] = "unsupported"

    def __init__(self) -> None:
        self.calls = 0

    def verify(self, rules: Sequence[Rule], *, domain_pack) -> list[VerificationResult]:
        self.calls += 1
        return [
            VerificationResult(
                rule_id=r.rule_id, supported=(r.effect.category != self._REJECT_CATEGORY),
                reason="ok" if r.effect.category != self._REJECT_CATEGORY else "unsupported by span",
            )
            for r in rules
        ]


def _corpus(md_book: Path) -> Corpus:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    return corpus


def test_build_rules_stores_only_verified_supported_rules(md_book: Path) -> None:
    corpus = _corpus(md_book)
    store = SqliteRuleStore(":memory:")
    detector = _FakeDetector()
    extractor = _FakeExtractor()
    verifier = _FakeVerifier()

    report = build_rules(corpus, store, detector, extractor, verifier, _FakeDomainPack())

    assert report.chapters_processed > 0
    assert report.rules_stored == report.rules_extracted
    assert report.rules_rejected_unsupported == 0
    assert len(store.rules_for_domain("test_domain")) == report.rules_stored


def test_an_unsupported_rule_is_never_stored(md_book: Path) -> None:
    corpus = _corpus(md_book)
    store = SqliteRuleStore(":memory:")
    detector = _FakeDetector()
    chunks = corpus.store.get_by_source(corpus.sources()[0].source_id)
    first_chunk_text = chunks[0].text
    extractor = _FakeExtractor(unsupported_category=first_chunk_text)
    verifier = _FakeVerifier()

    report = build_rules(corpus, store, detector, extractor, verifier, _FakeDomainPack())

    assert report.rules_rejected_unsupported >= 1
    assert report.rules_stored == report.rules_extracted - report.rules_rejected_unsupported
    all_stored = store.rules_for_domain("test_domain")
    assert all(r.effect.category != "unsupported" for r in all_stored)


def test_resume_skips_a_chapter_whose_fingerprint_already_matches(md_book: Path) -> None:
    corpus = _corpus(md_book)
    store = SqliteRuleStore(":memory:")
    detector = _FakeDetector()
    extractor = _FakeExtractor()
    verifier = _FakeVerifier()

    report1 = build_rules(corpus, store, detector, extractor, verifier, _FakeDomainPack())
    assert report1.chapters_processed > 0
    calls_after_first = len(detector.calls)

    report2 = build_rules(corpus, store, detector, extractor, verifier, _FakeDomainPack())

    assert report2.chapters_processed == 0
    assert report2.chapters_skipped == report1.chapters_processed
    assert len(detector.calls) == calls_after_first  # no re-detection


def test_a_chunk_with_low_quality_ocr_text_is_excluded_and_reported() -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    from ragcore.types import ChunkMeta

    def _mk_chunk(idx: int, text: str) -> Chunk:
        meta = ChunkMeta(
            source_id="src", source_path="book.md", source_title="Book", source_type="md",
            chapter_index=0, chapter_title="Ch 1", section=None, section_path=(), page_start=None,
            page_end=None, chunk_index=idx, context_group_id="src:0", char_start=0,
            char_end=len(text), token_count=10,
        )
        return Chunk(chunk_id=f"src:{idx}", text=text, vector=[0.0] * 8, meta=meta)

    garbled = "�" * 25
    good = "Mars in the eighth house causes health troubles during Mars dasha for a long time."
    corpus.store.upsert([_mk_chunk(0, garbled), _mk_chunk(1, good)])

    store = SqliteRuleStore(":memory:")
    detector = _FakeDetector()
    extractor = _FakeExtractor()
    verifier = _FakeVerifier()

    report = build_rules(corpus, store, detector, extractor, verifier, _FakeDomainPack())

    assert report.chunks_ocr_flagged == 1
    # the clean chunk was still processed
    assert detector.calls == [["src:1"]]


def test_a_chapter_entirely_ocr_flagged_is_skipped_without_calling_the_detector() -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    from ragcore.types import ChunkMeta

    def _mk_chunk(idx: int, text: str) -> Chunk:
        meta = ChunkMeta(
            source_id="src", source_path="book.md", source_title="Book", source_type="md",
            chapter_index=0, chapter_title="Ch 1", section=None, section_path=(), page_start=None,
            page_end=None, chunk_index=idx, context_group_id="src:0", char_start=0,
            char_end=len(text), token_count=10,
        )
        return Chunk(chunk_id=f"src:{idx}", text=text, vector=[0.0] * 8, meta=meta)

    garbled = "�" * 25
    corpus.store.upsert([_mk_chunk(0, garbled)])

    store = SqliteRuleStore(":memory:")
    detector = _FakeDetector()
    extractor = _FakeExtractor()
    verifier = _FakeVerifier()

    report = build_rules(corpus, store, detector, extractor, verifier, _FakeDomainPack())

    assert report.chapters_skipped_all_ocr_flagged == 1
    assert detector.calls == []


def test_a_failing_chapter_records_an_error_and_does_not_abort_other_chapters(md_book: Path) -> None:
    corpus = _corpus(md_book)
    store = SqliteRuleStore(":memory:")

    class _BoomDetector(_FakeDetector):
        def detect(self, chunks, *, domain_pack):
            self.calls.append([c.chunk_id for c in chunks])
            if len(self.calls) == 1:
                raise RuntimeError("boom")
            return super().detect(chunks, domain_pack=domain_pack)

    detector = _BoomDetector()
    extractor = _FakeExtractor()
    verifier = _FakeVerifier()

    report = build_rules(corpus, store, detector, extractor, verifier, _FakeDomainPack())

    assert len(report.errors) == 1
    assert "boom" in report.errors[0][2]
    assert report.chapters_processed >= 1  # the second chapter still succeeded
