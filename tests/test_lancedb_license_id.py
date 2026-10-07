"""ChunkMeta.license_id end to end through LanceDBStore: schema support,
filterability, and transparent migration of a table created before the
column existed — mirrors test_lancedb_tenant_id.py's shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("lancedb")
import lancedb as lancedb_module
import pyarrow as pa

from ragcore.stores.lancedb import LanceDBStore, _build_schema
from ragcore.types import Chunk, ChunkMeta


def _chunk(chunk_id: str, text: str, vector: list[float], license_id: str | None) -> Chunk:
    meta = ChunkMeta(
        source_id="s", source_path="p", source_title="t", source_type="md",
        section_path=(), chapter_title=None, chapter_index=None, section=None,
        page_start=None, page_end=None, chunk_index=0, context_group_id="g",
        char_start=0, char_end=len(text), token_count=1, license_id=license_id,
    )
    return Chunk(chunk_id=chunk_id, text=text, meta=meta, vector=vector)


def test_license_id_round_trips_through_upsert_and_query(tmp_path: Path) -> None:
    store = LanceDBStore(tmp_path / "db.lancedb", dim=2)
    store.upsert([_chunk("a", "hello", [1.0, 0.0], license_id="CC-BY-4.0")])

    [chunk] = store.query([1.0, 0.0], k=1)
    assert chunk.chunk.meta.license_id == "CC-BY-4.0"


def test_filtering_by_license_id(tmp_path: Path) -> None:
    store = LanceDBStore(tmp_path / "db.lancedb", dim=2)
    store.upsert(
        [
            _chunk("a", "open text", [1.0, 0.0], license_id="CC-BY-4.0"),
            _chunk("b", "closed text", [1.0, 0.0], license_id="proprietary"),
        ]
    )

    results = store.query([1.0, 0.0], k=10, filters={"license_id": "CC-BY-4.0"})

    assert {r.chunk.chunk_id for r in results} == {"a"}


def test_opening_a_pre_license_id_table_migrates_transparently(tmp_path: Path) -> None:
    db_path = tmp_path / "db.lancedb"
    old_schema = pa.schema(
        [f for f in _build_schema(dim=2) if f.name not in ("tenant_id", "license_id")]
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
    assert "license_id" not in old_table.schema.names

    store = LanceDBStore(db_path, dim=2)  # must migrate, not raise

    [chunk] = store.query([1.0, 0.0], k=1)
    assert chunk.chunk.chunk_id == "pre-existing"
    assert chunk.chunk.meta.license_id is None

    store.upsert([_chunk("new", "world", [0.0, 1.0], license_id="public-domain")])
    results = {r.chunk.chunk_id: r.chunk.meta.license_id for r in store.query([0.0, 1.0], k=10)}
    assert results == {"pre-existing": None, "new": "public-domain"}
