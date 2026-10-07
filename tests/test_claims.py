"""build_claims: chapter grouping, concept-menu anchoring, hallucinated-index
handling, the empty-concept-menu-skips-with-no-checkpoint rule, and
resumability (mirroring test_concept_resume.py's shape).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ragcore.claims.claude import _claim_id
from ragcore.claims.pipeline import build_claims
from ragcore.claims.sqlite_store import SqliteClaimStore
from ragcore.claims.types import SourceClaim
from ragcore.concepts.pipeline import build_concepts
from ragcore.concepts.sqlite_store import SqliteConceptStore
from ragcore.concepts.types import Concept
from ragcore.corpus import Corpus
from ragcore.types import Chunk
from tests.fakes import FakeEmbedder, InMemoryStore


class _FakeConceptExtractor:
    model = "fake-concepts"

    def extract(self, chunks: Sequence[Chunk]):
        concept = Concept(
            concept_id="concept:shared-idea", label="a shared idea", description="d",
            source_chunk_ids=tuple(c.chunk_id for c in chunks),
        )
        return [concept], []


class _FakeClaimExtractor:
    model = "fake-claims"

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str]]] = []

    def extract(self, *, chunks: Sequence[Chunk], concepts):
        self.calls.append(([c.chunk_id for c in chunks], [c.concept_id for c in concepts]))
        chunk = chunks[0]
        text = chunk.text.strip().splitlines()[0]
        return [
            SourceClaim(
                claim_id=_claim_id(chunk.meta.source_id, chunk.chunk_id, text),
                concept_ids=tuple(c.concept_id for c in concepts),
                source_id=chunk.meta.source_id, chunk_id=chunk.chunk_id, text=text,
                author_confidence="firm", model=self.model, prompt_version="v1", created_at="now",
                tenant_id=chunk.meta.tenant_id,
            )
        ]


def _corpus(md_book: Path) -> Corpus:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    return corpus


def test_build_claims_anchors_claims_to_the_chapters_own_concepts(md_book: Path) -> None:
    corpus = _corpus(md_book)
    concept_store = SqliteConceptStore(":memory:")
    build_concepts(corpus, concept_store, _FakeConceptExtractor())

    claim_store = SqliteClaimStore(":memory:")
    extractor = _FakeClaimExtractor()
    report = build_claims(corpus, claim_store, extractor, concept_store)

    assert report.claims_extracted == report.chapters_processed
    assert report.chapters_skipped_no_concepts == 0
    for _chunk_ids, concept_ids in extractor.calls:
        assert concept_ids == ["concept:shared-idea"]


def test_a_chapter_with_no_concepts_yet_is_skipped_with_no_checkpoint(md_book: Path) -> None:
    corpus = _corpus(md_book)
    concept_store = SqliteConceptStore(":memory:")  # never populated
    claim_store = SqliteClaimStore(":memory:")
    extractor = _FakeClaimExtractor()

    report = build_claims(corpus, claim_store, extractor, concept_store)

    assert report.claims_extracted == 0
    assert report.chapters_skipped_no_concepts > 0
    assert extractor.calls == []  # never even called — no concepts to anchor to

    # No checkpoint written means concepts arriving later lets a re-run pick
    # the chapter up "for free".
    build_concepts(corpus, concept_store, _FakeConceptExtractor())
    report2 = build_claims(corpus, claim_store, extractor, concept_store)
    assert report2.claims_extracted > 0


def test_resume_skips_a_chapter_whose_fingerprint_already_matches(md_book: Path) -> None:
    corpus = _corpus(md_book)
    concept_store = SqliteConceptStore(":memory:")
    build_concepts(corpus, concept_store, _FakeConceptExtractor())

    claim_store = SqliteClaimStore(":memory:")
    extractor = _FakeClaimExtractor()

    report1 = build_claims(corpus, claim_store, extractor, concept_store)
    calls_after_first = len(extractor.calls)
    assert report1.chapters_processed > 0

    report2 = build_claims(corpus, claim_store, extractor, concept_store)
    assert report2.chapters_processed == 0
    assert report2.chapters_skipped == report1.chapters_processed
    assert len(extractor.calls) == calls_after_first  # no re-extraction


def test_a_concept_id_change_invalidates_the_checkpoint(md_book: Path) -> None:
    """build_concepts' resolver can remap a chapter's concept ids between
    runs even when its chunks haven't changed — the claims fingerprint must
    include concept_ids, not just chunk_ids, so a stale claim never stays
    checkpointed against a concept id that no longer exists."""
    corpus = _corpus(md_book)
    concept_store = SqliteConceptStore(":memory:")
    build_concepts(corpus, concept_store, _FakeConceptExtractor())

    claim_store = SqliteClaimStore(":memory:")
    extractor = _FakeClaimExtractor()
    build_claims(corpus, claim_store, extractor, concept_store)
    calls_after_first = len(extractor.calls)

    # Simulate a concept resolver remap: same chunks, different concept id.
    concept_store2 = SqliteConceptStore(":memory:")

    class _RenamedConceptExtractor(_FakeConceptExtractor):
        def extract(self, chunks: Sequence[Chunk]):
            concept = Concept(
                concept_id="concept:renamed-idea", label="a shared idea", description="d",
                source_chunk_ids=tuple(c.chunk_id for c in chunks),
            )
            return [concept], []

    build_concepts(corpus, concept_store2, _RenamedConceptExtractor())
    report2 = build_claims(corpus, claim_store, extractor, concept_store2)

    assert report2.chapters_skipped == 0  # different concept_ids -> different fingerprint
    assert len(extractor.calls) > calls_after_first


def test_an_extractor_returning_no_claims_is_tolerated(md_book: Path) -> None:
    """A ClaimExtractor implementation may legitimately decide a chapter has
    no genuine, concept-anchored claims (e.g. every citation it considered
    was hallucinated) — build_claims must not choke on an empty result and
    must still checkpoint the chapter as done."""
    corpus = _corpus(md_book)
    concept_store = SqliteConceptStore(":memory:")
    build_concepts(corpus, concept_store, _FakeConceptExtractor())

    class _EmptyExtractor:
        model = "fake-empty"

        def extract(self, *, chunks, concepts):
            return []

    claim_store = SqliteClaimStore(":memory:")
    report = build_claims(corpus, claim_store, _EmptyExtractor(), concept_store)

    assert report.claims_extracted == 0
    assert report.chapters_processed > 0
    assert report.errors == ()
    for source in corpus.sources():
        for chunk in corpus.store.get_by_source(source.source_id):
            assert claim_store.checkpoint(source.source_id, chunk.meta.context_group_id) is not None


def test_build_claims_persists_the_chunks_own_tenant_id(md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book, tenant_id="tenant-a")
    concept_store = SqliteConceptStore(":memory:")
    build_concepts(corpus, concept_store, _FakeConceptExtractor())

    claim_store = SqliteClaimStore(":memory:")
    build_claims(corpus, claim_store, _FakeClaimExtractor(), concept_store)

    claims = claim_store.claims_for_concept("concept:shared-idea")
    assert claims
    assert all(c.tenant_id == "tenant-a" for c in claims)


def test_opening_a_pre_tenant_id_claims_table_migrates_via_alter_table(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "claims.sqlite"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE claims (claim_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, "
        "chunk_id TEXT NOT NULL, text TEXT NOT NULL, author_confidence TEXT NOT NULL, "
        "model TEXT NOT NULL, prompt_version TEXT NOT NULL, created_at TEXT NOT NULL, "
        "vector_json TEXT)"
    )
    conn.execute(
        "INSERT INTO claims (claim_id, source_id, chunk_id, text, author_confidence, model, "
        "prompt_version, created_at, vector_json) "
        "VALUES ('c1', 's1', 'ch1', 'text', 'firm', 'm', 'v1', 'now', NULL)"
    )
    conn.commit()
    conn.close()

    store = SqliteClaimStore(path)  # must migrate via ALTER TABLE, not raise

    claim = store.claim("c1")
    assert claim is not None
    assert claim.tenant_id is None
