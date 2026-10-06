"""ChunkMeta.tenant_id end to end through LanceDBStore: schema support,
filterability, and transparent migration of a table created before the
column existed (docs/adr/0006-tenant-id-foundation-not-enforcement.md).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("lancedb")
import lancedb as lancedb_module
import pyarrow as pa

from ragcore.stores.lancedb import LanceDBStore, _build_schema
from ragcore.types import Chunk, ChunkMeta


def _chunk(chunk_id: str, text: str, vector: list[float], tenant_id: str | None) -> Chunk:
    meta = ChunkMeta(
        source_id="s", source_path="p", source_title="t", source_type="md",
        section_path=(), chapter_title=None, chapter_index=None, section=None,
        page_start=None, page_end=None, chunk_index=0, context_group_id="g",
        char_start=0, char_end=len(text), token_count=1, tenant_id=tenant_id,
    )
    return Chunk(chunk_id=chunk_id, text=text, meta=meta, vector=vector)


def test_tenant_id_round_trips_through_upsert_and_query(tmp_path: Path) -> None:
    store = LanceDBStore(tmp_path / "db.lancedb", dim=2)
    store.upsert([_chunk("a", "hello", [1.0, 0.0], tenant_id="tenant-a")])

    [chunk] = store.query([1.0, 0.0], k=1)
    assert chunk.chunk.meta.tenant_id == "tenant-a"


def test_tenant_id_defaults_to_none_when_not_set(tmp_path: Path) -> None:
    store = LanceDBStore(tmp_path / "db.lancedb", dim=2)
    store.upsert([_chunk("a", "hello", [1.0, 0.0], tenant_id=None)])

    [chunk] = store.query([1.0, 0.0], k=1)
    assert chunk.chunk.meta.tenant_id is None


def test_filtering_by_tenant_id_isolates_tenants(tmp_path: Path) -> None:
    store = LanceDBStore(tmp_path / "db.lancedb", dim=2)
    store.upsert(
        [
            _chunk("a", "hello from tenant a", [1.0, 0.0], tenant_id="tenant-a"),
            _chunk("b", "hello from tenant b", [1.0, 0.0], tenant_id="tenant-b"),
        ]
    )

    results = store.query([1.0, 0.0], k=10, filters={"tenant_id": "tenant-a"})

    assert {r.chunk.chunk_id for r in results} == {"a"}


def test_opening_a_pre_tenant_id_table_migrates_the_schema_transparently(tmp_path: Path) -> None:
    """Simulates an on-disk LanceDB table created before tenant_id existed
    on ChunkMeta, by hand-building the old schema shape (every field
    _build_schema has today except tenant_id) and writing a row through
    it directly, bypassing LanceDBStore entirely. Opening it afterward
    through the current LanceDBStore must not raise, and the pre-existing
    row must read back with tenant_id=None rather than erroring."""
    db_path = tmp_path / "db.lancedb"
    old_schema = pa.schema(
        [f for f in _build_schema(dim=2) if f.name != "tenant_id"]
    )
    db = lancedb_module.connect(str(db_path))
    old_table = db.create_table("chunks", schema=old_schema)
    old_table.add(
        [
            {
                "chunk_id": "pre-existing",
                "text": "hello",
                "vector": [1.0, 0.0],
                "source_id": "s",
                "source_path": "p",
                "source_title": "t",
                "source_type": "md",
                "section_path": [],
                "chapter_title": None,
                "chapter_index": None,
                "section": None,
                "page_start": None,
                "page_end": None,
                "chunk_index": 0,
                "context_group_id": "g",
                "char_start": 0,
                "char_end": 5,
                "token_count": 1,
                "extra_json": "{}",
            }
        ]
    )
    assert "tenant_id" not in old_table.schema.names

    store = LanceDBStore(db_path, dim=2)  # must migrate, not raise

    [chunk] = store.query([1.0, 0.0], k=1)
    assert chunk.chunk.chunk_id == "pre-existing"
    assert chunk.chunk.meta.tenant_id is None

    store.upsert([_chunk("new", "world", [0.0, 1.0], tenant_id="tenant-a")])
    results = {r.chunk.chunk_id: r.chunk.meta.tenant_id for r in store.query([0.0, 1.0], k=10)}
    assert results == {"pre-existing": None, "new": "tenant-a"}
