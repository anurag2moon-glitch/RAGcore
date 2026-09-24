"""Requirement 3: concurrent ingestion — simulate multiple books being
ingested at the same time; verify no data corruption, no partial writes,
no race conditions in LanceDBStore and SqliteGraphStore.

Also the direct regression tests for the two thread-safety defects fixed
in ragcore/graph/sqlite_store.py (see docs/INTEGRATION_TEST_FINDINGS.md):
`check_same_thread=True` made any cross-thread use raise outright, and the
add_entities/add_relations read-then-write merge had no transaction guard
(a lost-update race under concurrent writers to the same entity/relation
id — the exact case concurrent multi-book ingestion into one shared graph
produces whenever two books mention the same cross-book entity "at the
same time").
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from ragcore.corpus import Corpus
from ragcore.graph.sqlite_store import SqliteGraphStore
from ragcore.graph.types import Entity
from tests.integration.doubles import DeterministicEmbedder
from tests.integration.synth.library import Library
from tests.fakes import InMemoryStore


def test_concurrent_ingest_many_no_partial_writes_no_duplicates(
    fresh_corpus: Corpus, synth_library: Library
) -> None:
    keys = [b.key for b in synth_library.books]  # all 8, ingested concurrently
    expected_chunks = {}
    # Pre-ingest sequentially into a throwaway corpus just to know the
    # expected per-book chunk count for the assertion below (ingestion is
    # deterministic — same file bytes always chunk identically).
    probe = Corpus(InMemoryStore(dim=8), DeterministicEmbedder(dim=8))
    for key in keys:
        r = probe.ingest(synth_library.book(key).path)
        expected_chunks[r.source_ids[0]] = r.chunks_ingested

    report = fresh_corpus.ingest_many(
        [synth_library.book(k).path for k in keys], max_workers=8
    )

    assert report.errors == []
    assert len(report.source_ids) == len(keys)
    assert len(set(report.source_ids)) == len(keys)  # no duplicate source_ids

    total = fresh_corpus.count()
    assert total == sum(expected_chunks.values())

    sources = {s.source_id: s.chunk_count for s in fresh_corpus.sources()}
    assert set(sources) == set(expected_chunks)
    for source_id, expected in expected_chunks.items():
        assert sources[source_id] == expected, f"partial write for {source_id}"
        # every one of that source's chunks is actually fetchable, not
        # just counted
        chunks = fresh_corpus.store.get_by_source(source_id)
        assert len(chunks) == expected
        assert len({c.chunk_id for c in chunks}) == expected  # no duplicate ids


def test_concurrent_ingest_from_separate_store_connections(
    tmp_path, synth_library: Library
) -> None:
    """Mirrors real multi-process ingestion: each worker opens its OWN
    LanceDBStore connection to the same on-disk table, rather than sharing
    one Corpus/store instance."""
    from ragcore.stores.lancedb import LanceDBStore

    db_path = str(tmp_path / "concurrent.lancedb")
    LanceDBStore(db_path, dim=8)  # create the table up front

    keys = [b.key for b in synth_library.books]
    errors: list[str] = []

    def worker(key: str) -> None:
        try:
            store = LanceDBStore(db_path, dim=8)
            corpus = Corpus(store, DeterministicEmbedder(dim=8))
            corpus.ingest(synth_library.book(key).path)
        except Exception as e:  # noqa: BLE001 - collected for the assertion below
            errors.append(f"{key}: {type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, keys))

    assert errors == []
    final_store = LanceDBStore(db_path, dim=8)
    sources = final_store.list_sources()
    assert len(sources) == len(keys)
    assert final_store.count() == sum(s.chunk_count for s in sources)


def test_concurrent_graph_writers_same_store_instance_no_lost_updates() -> None:
    """Regression test for the check_same_thread + read-modify-write race:
    many threads sharing ONE SqliteGraphStore instance all merge into the
    SAME entity_id concurrently — every writer's chunk id and alias must
    survive (a lost update would silently drop some)."""
    graph = SqliteGraphStore(":memory:")
    n_writers = 40

    def worker(i: int) -> None:
        graph.add_entities(
            [
                Entity(
                    entity_id="entity:person:shared",
                    name="Shared",
                    entity_type="person",
                    source_chunk_ids=(f"chunk-{i}",),
                    extra={"aliases": [f"alias-{i}"]},
                )
            ]
        )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_writers)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    entities = graph.entities_by_type("person")
    assert len(entities) == 1
    merged = entities[0]
    assert len(merged.source_chunk_ids) == n_writers, "lost update: some writers' chunk ids missing"
    assert len(merged.extra.get("aliases", [])) == n_writers, "lost update: some writers' aliases missing"


def test_concurrent_graph_writers_separate_connections_same_file(tmp_path) -> None:
    """Cross-process-style concurrency: each thread opens its OWN
    SqliteGraphStore connection to the same file. Exercises the BEGIN
    IMMEDIATE + busy_timeout path (WAL + threading.Lock alone wouldn't
    help here, since each connection has its own lock object)."""
    db_path = str(tmp_path / "concurrent_graph.sqlite3")
    SqliteGraphStore(db_path)  # create schema up front
    n_writers = 20
    errors: list[str] = []

    def worker(i: int) -> None:
        try:
            g = SqliteGraphStore(db_path)
            g.add_entities(
                [
                    Entity(
                        entity_id="entity:person:shared",
                        name="Shared",
                        entity_type="person",
                        source_chunk_ids=(f"chunk-{i}",),
                    )
                ]
            )
        except Exception as e:  # noqa: BLE001
            errors.append(f"{i}: {type(e).__name__}: {e}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_writers)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert errors == []
    final = SqliteGraphStore(db_path)
    entities = final.entities_by_type("person")
    assert len(entities) == 1
    assert len(entities[0].source_chunk_ids) == n_writers


def test_interleaved_ingest_and_query_no_crash_no_stale_corruption(
    fresh_corpus: Corpus, synth_library: Library
) -> None:
    """Queries run concurrently with ingestion of other books — a reader
    should never see a torn/partial write, and ingestion should complete
    cleanly regardless of concurrent read load."""
    keys = [b.key for b in synth_library.books]
    stop = threading.Event()
    query_errors: list[str] = []

    def query_loop() -> None:
        while not stop.is_set():
            try:
                fresh_corpus.retrieve("the captain sighted the tide charts", k=5)
            except Exception as e:  # noqa: BLE001
                query_errors.append(f"{type(e).__name__}: {e}")

    readers = [threading.Thread(target=query_loop) for _ in range(3)]
    [t.start() for t in readers]
    try:
        report = fresh_corpus.ingest_many([synth_library.book(k).path for k in keys], max_workers=4)
    finally:
        stop.set()
        [t.join() for t in readers]

    assert report.errors == []
    assert query_errors == []
    assert fresh_corpus.count() == sum(s.chunk_count for s in fresh_corpus.sources())
