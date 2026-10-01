"""Requirement 5: failure injection — kill ingestion mid-file, corrupt an
index file, simulate running out of disk space, feed malformed input files.
Verify ragcore fails gracefully (a typed RagcoreError, not a bare crash)
and — the important part — never corrupts or loses data that was already
successfully committed before the injected failure.
"""

from __future__ import annotations

import os

import pytest

from ragcore.corpus import Corpus
from ragcore.errors import RagcoreError, StoreError
from ragcore.graph.sqlite_store import SqliteGraphStore
from ragcore.graph.types import Entity
from ragcore.stores.lancedb import LanceDBStore
from tests.integration.doubles import DeterministicEmbedder, FailingEmbedder, FailingStore
from tests.integration.synth.library import Library


def test_embedding_failure_mid_document_leaves_no_partial_chunks(
    fresh_corpus: Corpus, synth_library: Library
) -> None:
    """A book that dies partway through embedding must not leave any of
    its chunks in the store — Corpus.ingest() only calls store.upsert()
    after embedding succeeds for the whole document, so this should
    already hold; this test locks that contract in."""
    # Corpus.ingest() calls self.embedder directly (not through
    # self.retriever, which only matters for retrieve()), so wrapping just
    # this attribute is sufficient to fail ingestion specifically.
    fresh_corpus.embedder = FailingEmbedder(fresh_corpus.embedder, fail_after_n_calls=1)

    spec = synth_library.book("clean_toc")
    with pytest.raises(RagcoreError):
        fresh_corpus.ingest(spec.path)

    assert fresh_corpus.count() == 0
    assert fresh_corpus.sources() == []


def test_embedding_failure_partway_through_ingest_many_keeps_healthy_books(
    fresh_corpus: Corpus, synth_library: Library
) -> None:
    """Simulates a process kill/network drop partway through a multi-book
    ingestion run: the Nth file's embedding call fails, but files ingested
    before and after (best-effort, ingest_many collects errors rather than
    aborting) keep their data intact."""
    keys = ["very_short", "unusual_naming", "clean_toc", "page_structured"]
    failing_embedder = FailingEmbedder(fresh_corpus.embedder, fail_after_n_calls=3)
    fresh_corpus.embedder = failing_embedder

    report = fresh_corpus.ingest_many(
        [synth_library.book(k).path for k in keys], max_workers=1
    )

    assert len(report.errors) >= 1, "expected at least one simulated failure to be recorded"
    failed_paths = {p for p, _ in report.errors}
    succeeded_keys = [k for k in keys if str(synth_library.book(k).path) not in failed_paths]
    assert succeeded_keys, "expected at least one book to survive the simulated failure"

    for source in fresh_corpus.sources():
        chunks = fresh_corpus.store.get_by_source(source.source_id)
        assert len(chunks) == source.chunk_count  # every surviving source is fully, not partially, present


def test_store_write_failure_during_ingest_leaves_no_partial_write(
    tmp_path, synth_library: Library
) -> None:
    """Simulated disk-full (ENOSPC) on upsert: the document must not
    appear in the store at all (Corpus.ingest() has already computed and
    embedded all chunks by the time upsert() is called; failure there
    should not leave a torn write for THIS document, and must not disturb
    anything ingested earlier)."""
    embedder = DeterministicEmbedder(dim=64)
    real_store = LanceDBStore(str(tmp_path / "corpus.lancedb"), dim=64)
    corpus = Corpus(real_store, embedder)

    healthy = synth_library.book("very_short")
    corpus.ingest(healthy.path)
    healthy_count = corpus.count()

    corpus.store = FailingStore(real_store, fail_on="upsert", fail_after_n_calls=1)
    victim = synth_library.book("unusual_naming")
    with pytest.raises(RagcoreError):
        corpus.ingest(victim.path)

    # the failed store wrapper delegates reads straight to the real store,
    # so this reflects the real store's actual state
    assert real_store.count() == healthy_count
    remaining_sources = {s.source_id for s in real_store.list_sources()}
    assert len(remaining_sources) == 1  # only the healthy book


def test_sqlite_graph_wal_sidecar_loss_raises_a_typed_store_error(tmp_path) -> None:
    """Enabling WAL mode means recent writes live in the `-wal`/`-shm`
    sidecar files until a checkpoint, not in the main `.sqlite3` file
    itself. Copying, backing up, or restoring only the main file (a real,
    easy operational mistake) silently loses everything not yet
    checkpointed, and reopening the orphaned main file raises a backend
    error — this test's job is to confirm that error surfaces as
    ragcore.errors.StoreError (via ragcore.errors.wrap_backend_errors, see
    docs/adr/0004-wrap-backend-exceptions-broadly.md), not a raw
    sqlite3.OperationalError, per ragcore/errors.py's own stated
    exception-hierarchy contract. Previously (docs/INTEGRATION_TEST_FINDINGS.md)
    this test documented the opposite, unwrapped behavior as a known gap;
    it now asserts the fixed behavior.
    """
    db_path = str(tmp_path / "graph.sqlite3")
    graph = SqliteGraphStore(db_path)
    graph.add_entities(
        [Entity(entity_id="entity:person:x", name="X", entity_type="person", source_chunk_ids=("c1",))]
    )
    del graph  # no explicit close/checkpoint — mirrors a killed process

    assert os.path.exists(db_path + "-wal"), "test premise: WAL sidecar must exist to demonstrate the gap"
    os.remove(db_path + "-wal")
    os.remove(db_path + "-shm")

    with pytest.raises(StoreError, match="disk I/O error"):
        SqliteGraphStore(db_path)


def test_malformed_inputs_raise_typed_errors_or_degrade_gracefully(fresh_corpus: Corpus, tmp_path) -> None:
    """Every format-loader failure must surface as a RagcoreError subclass
    (LoaderError), never a bare native exception. Trivially-empty inputs
    are allowed to degrade gracefully (0 chunks) rather than error, per
    each loader's own documented policy — this test asserts the *typed
    or graceful* dichotomy, not that everything must raise."""
    cases: dict[str, bytes] = {
        "empty.txt": b"",
        "empty.md": b"",
        "bad_utf8.txt": b"\xff\xfe\x00bad-but-not-fatal",
        "fake.epub": b"not a zip file at all",
        "fake.pdf": b"%PDF-garbage-not-a-real-pdf-stream",
        "fake.docx": b"not a real docx either",
        "unknown_ext.xyz": b"hello world",
    }
    for name, content in cases.items():
        path = tmp_path / name
        path.write_bytes(content)
        try:
            report = fresh_corpus.ingest(path)
            assert report.chunks_ingested >= 0  # graceful path: no crash, a real (possibly empty) report
        except RagcoreError:
            pass  # typed failure: acceptable
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"{name}: raised a non-RagcoreError exception: {type(e).__name__}: {e}")


def test_directory_path_raises_typed_error(fresh_corpus: Corpus, tmp_path) -> None:
    a_dir = tmp_path / "not_a_file"
    a_dir.mkdir()
    with pytest.raises(RagcoreError):
        fresh_corpus.ingest(a_dir)
    assert fresh_corpus.count() == 0


def test_zero_chapter_book_ingests_as_empty_or_single_block_without_crashing(
    fresh_corpus: Corpus, tmp_path
) -> None:
    """An .md file with no headings at all and no body — the degenerate
    "book" — should not crash the pipeline; 0 chunks is an acceptable
    outcome."""
    path = tmp_path / "blank.md"
    path.write_text("", encoding="utf-8")
    report = fresh_corpus.ingest(path)
    assert report.chunks_ingested == 0
    assert fresh_corpus.count() == 0


def test_healthy_ingestion_survives_after_a_prior_failed_ingest_in_the_same_corpus(
    fresh_corpus: Corpus, synth_library: Library, tmp_path
) -> None:
    """The most important recovery property: after ANY of the failures
    above, the corpus must still be usable for the NEXT, healthy ingest —
    a failed ingest must not leave the store or corpus object in a state
    that poisons subsequent operations."""
    bad_path = tmp_path / "fake.epub"
    bad_path.write_bytes(b"not a zip file")
    with pytest.raises(RagcoreError):
        fresh_corpus.ingest(bad_path)

    healthy = synth_library.book("very_short")
    report = fresh_corpus.ingest(healthy.path)
    assert report.chunks_ingested > 0
    assert fresh_corpus.count() == report.chunks_ingested
    results = fresh_corpus.retrieve("a", k=5)
    assert isinstance(results, list)
