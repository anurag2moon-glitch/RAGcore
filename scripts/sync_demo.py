"""Demonstrates incremental ingestion + rollback (ragcore.versioning):

    1. Ingest a two-book corpus, tracked by an IngestionManifest.
    2. Change one book's content on disk.
    3. Run sync_corpus() and confirm: the unchanged book's graph/concept
       rows are untouched, the changed book's OLD chunks are gone (not
       orphaned — see ragcore/versioning/sync.py's module docstring for the
       bug this works around) and its NEW content is indexed under a new
       source_id.
    4. Roll back to the pre-sync snapshot and confirm the vector store and
       every SQLite-backed store are back to their pre-sync state.

Uses a real on-disk LanceDBStore (snapshot/restore needs LanceDB's native
versioning, which InMemoryStore doesn't have) but fakes every LLM-backed
component — no API calls. Everything lives under a temp directory that's
cleaned up at the end.

Usage:
    python scripts/sync_demo.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Sequence

from ragcore.corpus import Corpus
from ragcore.graph.sqlite_store import SqliteGraphStore
from ragcore.graph.types import Entity, Relation
from ragcore.stores.lancedb import LanceDBStore
from ragcore.types import Chunk
from ragcore.versioning.manifest import IngestionManifest
from ragcore.versioning.snapshot import rollback
from ragcore.versioning.sync import sync_corpus
from tests.fakes import FakeEmbedder

BOOK_A_V1 = """\
# Chapter 1: Steady Ground

Book A discusses the virtue of patience in long endeavors.
"""

BOOK_A_V2 = """\
# Chapter 1: Steady Ground

Book A now discusses the virtue of patience, and also of courage, in long
endeavors.
"""

BOOK_B = """\
# Chapter 1: A Different Matter

Book B discusses the discipline required to keep a promise.
"""


class _FakeEntityExtractor:
    model = "fake-entities"

    def extract(self, chunks: Sequence[Chunk]) -> tuple[list[Entity], list[Relation]]:
        text = " ".join(c.text for c in chunks).lower()
        entities = []
        if "patience" in text:
            entities.append(
                Entity(entity_id="entity:concept:patience", name="Patience", entity_type="concept",
                       source_chunk_ids=tuple(c.chunk_id for c in chunks))
            )
        if "courage" in text:
            entities.append(
                Entity(entity_id="entity:concept:courage", name="Courage", entity_type="concept",
                       source_chunk_ids=tuple(c.chunk_id for c in chunks))
            )
        if "promise" in text:
            entities.append(
                Entity(entity_id="entity:concept:promise", name="Promise", entity_type="concept",
                       source_chunk_ids=tuple(c.chunk_id for c in chunks))
            )
        return entities, []


def _print_state(label: str, corpus: Corpus, graph_store: SqliteGraphStore, manifest: IngestionManifest) -> None:
    print(f"\n--- {label} ---")
    print(f"  vector store chunk count: {corpus.count()}")
    for s in sorted(corpus.sources(), key=lambda s: s.path or ""):
        print(f"    source: {s.path}  chunks={s.chunk_count}  source_id={s.source_id[:12]}...")
    entity_names = sorted(e.name for e in graph_store.entities_by_type("concept"))
    print(f"  graph entities: {entity_names}")
    print(f"  manifest tracks: {sorted(Path(p).name for p in manifest.current_state())}")


def main() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="ragcore_sync_demo_"))
    try:
        book_a_path = tmp_dir / "book_a.md"
        book_b_path = tmp_dir / "book_b.md"
        book_a_path.write_text(BOOK_A_V1)
        book_b_path.write_text(BOOK_B)

        embedder = FakeEmbedder(dim=16)
        store = LanceDBStore(tmp_dir / "vectors", dim=16)
        corpus = Corpus(store, embedder)

        graph_store = SqliteGraphStore(tmp_dir / "graph.sqlite3")
        manifest = IngestionManifest(tmp_dir / "manifest.sqlite3")
        snapshot_dir = tmp_dir / "snapshots"

        print("=== Initial sync: ingesting book A (v1) and book B ===")
        report1 = sync_corpus(
            corpus, manifest, [book_a_path, book_b_path],
            graph_store=graph_store, graph_extractor=_FakeEntityExtractor(),
            snapshot_dir=snapshot_dir,
        )
        print(f"  added={len(report1.added)} changed={len(report1.changed)} "
              f"removed={len(report1.removed)} chunks_ingested={report1.chunks_ingested}")
        _print_state("after initial sync", corpus, graph_store, manifest)

        print("\n=== Editing book A on disk (adds 'courage') ===")
        book_a_path.write_text(BOOK_A_V2)

        print("\n=== Second sync: only book A changed ===")
        report2 = sync_corpus(
            corpus, manifest, [book_a_path, book_b_path],
            graph_store=graph_store, graph_extractor=_FakeEntityExtractor(),
            snapshot_dir=snapshot_dir,
        )
        print(f"  added={len(report2.added)} changed={len(report2.changed)} "
              f"removed={len(report2.removed)} chunks_ingested={report2.chunks_ingested}")
        print(f"  pre-sync snapshot tag: {report2.snapshot_tag}")
        _print_state("after second sync (book A changed)", corpus, graph_store, manifest)
        print(
            "  -> book A's OLD chunks are gone (not orphaned), 'courage' now shows up "
            "in the graph, and book B's entities were never touched by this sync."
        )

        print("\n=== Rolling back to the snapshot taken just before the second sync ===")
        rollback(
            manifest, report2.snapshot_tag, corpus.store,
            graph_store=graph_store,
        )
        _print_state("after rollback", corpus, graph_store, manifest)
        print("  -> back to book A (v1) + book B, matching the state right after the first sync.")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
