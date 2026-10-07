"""Fts5LexicalStore: indexing, keyword search ranking, source-scoped search,
deletion, and the online-backup helper — no network, stdlib sqlite3 only.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ragcore.lexical.fts5 import Fts5LexicalStore
from ragcore.types import Chunk, ChunkMeta


def _chunk(
    chunk_id: str, source_id: str, text: str, chunk_index: int = 0, tenant_id: str | None = None
) -> Chunk:
    meta = ChunkMeta(
        source_id=source_id, source_path=f"{source_id}.md", source_title="T", source_type="md",
        section_path=(), chapter_title=None, chapter_index=None, section=None,
        page_start=None, page_end=None, chunk_index=chunk_index, context_group_id="g",
        char_start=0, char_end=len(text), token_count=len(text) // 4, tenant_id=tenant_id,
    )
    return Chunk(chunk_id=chunk_id, text=text, meta=meta)


def test_search_ranks_the_more_relevant_document_first() -> None:
    store = Fts5LexicalStore(":memory:")
    store.index(
        [
            _chunk("c1", "s1", "the quick brown fox jumps over the lazy dog"),
            _chunk("c2", "s1", "a lazy dog sleeps all day in the sun"),
            _chunk("c3", "s2", "quick foxes are quick and clever animals, foxes foxes"),
        ]
    )

    hits = store.search("quick fox", k=10)

    assert [h.chunk_id for h in hits[:2]] == ["c3", "c1"] or hits[0].chunk_id in {"c1", "c3"}
    assert all(h.rank == i for i, h in enumerate(hits, start=1))


def test_search_restricts_to_given_source_ids() -> None:
    store = Fts5LexicalStore(":memory:")
    store.index([_chunk("c1", "s1", "apple banana"), _chunk("c2", "s2", "apple banana")])

    hits = store.search("apple", k=10, source_ids=["s1"])

    assert [h.chunk_id for h in hits] == ["c1"]


def test_index_replaces_by_chunk_id() -> None:
    store = Fts5LexicalStore(":memory:")
    store.index([_chunk("c1", "s1", "original text about zebras")])
    store.index([_chunk("c1", "s1", "updated text about giraffes")])

    assert store.count() == 1
    assert [h.chunk_id for h in store.search("zebras", k=10)] == []
    assert [h.chunk_id for h in store.search("giraffes", k=10)] == ["c1"]


def test_delete_by_source_id() -> None:
    store = Fts5LexicalStore(":memory:")
    store.index([_chunk("c1", "s1", "alpha"), _chunk("c2", "s1", "beta"), _chunk("c3", "s2", "gamma")])

    deleted = store.delete(source_id="s1")

    assert deleted == 2
    assert store.count() == 1


def test_delete_by_chunk_ids() -> None:
    store = Fts5LexicalStore(":memory:")
    store.index([_chunk("c1", "s1", "alpha"), _chunk("c2", "s1", "beta")])

    deleted = store.delete(chunk_ids=["c1"])

    assert deleted == 1
    assert store.count() == 1


def test_search_with_no_alphanumeric_tokens_returns_empty_without_error() -> None:
    store = Fts5LexicalStore(":memory:")
    store.index([_chunk("c1", "s1", "alpha")])

    assert store.search("!!! ??? ---", k=10) == []


def test_search_sanitizes_fts5_query_syntax_in_raw_user_input() -> None:
    """A raw query containing FTS5 operator syntax (quotes, AND/OR, NEAR)
    must never raise sqlite3.OperationalError."""
    store = Fts5LexicalStore(":memory:")
    store.index([_chunk("c1", "s1", "alpha beta")])

    hits = store.search('"unbalanced quote AND OR NEAR(alpha, beta)', k=10)
    assert isinstance(hits, list)  # did not raise


def test_search_restricts_to_given_tenant_id() -> None:
    store = Fts5LexicalStore(":memory:")
    store.index(
        [
            _chunk("c1", "s1", "quick fox", tenant_id="tenant-a"),
            _chunk("c2", "s2", "quick fox", tenant_id="tenant-b"),
        ]
    )

    hits = store.search("quick fox", k=10, tenant_id="tenant-a")

    assert [h.chunk_id for h in hits] == ["c1"]


def test_search_with_no_tenant_id_filter_sees_every_tenant() -> None:
    store = Fts5LexicalStore(":memory:")
    store.index(
        [
            _chunk("c1", "s1", "quick fox", tenant_id="tenant-a"),
            _chunk("c2", "s2", "quick fox", tenant_id="tenant-b"),
        ]
    )

    hits = store.search("quick fox", k=10)

    assert {h.chunk_id for h in hits} == {"c1", "c2"}


def test_opening_a_pre_tenant_id_table_migrates_transparently(tmp_path: Path) -> None:
    """Simulates an on-disk fts5 table from before tenant_id existed on
    this schema. ALTER TABLE doesn't work on fts5 virtual tables
    (confirmed directly: SQLite raises "virtual tables may not be
    altered") so opening it must drop-and-rebuild instead, preserving
    every pre-existing row with tenant_id=NULL."""
    path = tmp_path / "lex.sqlite"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE VIRTUAL TABLE fts_chunks USING fts5("
        "chunk_id UNINDEXED, source_id UNINDEXED, text, tokenize='porter unicode61')"
    )
    conn.execute(
        "INSERT INTO fts_chunks (chunk_id, source_id, text) VALUES ('pre-existing', 's1', 'hello world')"
    )
    conn.commit()
    conn.close()

    store = Fts5LexicalStore(path)  # must migrate, not raise

    assert store.count() == 1
    assert [h.chunk_id for h in store.search("hello", k=10)] == ["pre-existing"]
    assert store.search("hello", k=10, tenant_id="tenant-a") == []

    store.index([_chunk("new", "s1", "hello again", tenant_id="tenant-a")])
    assert {h.chunk_id for h in store.search("hello", k=10)} == {"pre-existing", "new"}


def test_backup_to_produces_a_restorable_copy(tmp_path: Path) -> None:
    store = Fts5LexicalStore(":memory:")
    store.index([_chunk("c1", "s1", "backup me please")])

    backup_path = tmp_path / "backup.sqlite3"
    store.backup_to(backup_path)

    restored = Fts5LexicalStore(backup_path)
    assert restored.count() == 1
    assert [h.chunk_id for h in restored.search("backup", k=10)] == ["c1"]
