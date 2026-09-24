"""VectorStore backed by LanceDB — embedded, file-based, zero-ops.

ChunkMeta's typed fields are persisted as real columns (so structural
filters like chapter_index stay indexable); `extra` is serialized to a JSON
string column, since its shape is caller-defined and not known up front.
`section_path` is stored as a string list column for round-tripping the
full breadcrumb, but is not itself filterable through the Filter DSL (use
`chapter`/`chapter_index`/`section`, the denormalized scalar fields).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from ragcore.errors import StoreError
from ragcore.types import Chunk, ChunkMeta, Filter, Retrieved, SourceInfo, Vector

try:
    import lancedb
    import pyarrow as pa
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    lancedb = None
    pa = None
else:
    _IMPORT_ERROR = None

# Scalar ChunkMeta fields exposed to the Filter DSL. section_path (a list
# column) and extra (opaque JSON) are deliberately excluded — filtering on
# those would require a different predicate grammar than the flat
# equality/membership/range DSL the rest of ragcore is written against.
# chapter_title is *not* filterable, on purpose: it's raw, possibly-noisy
# display text (see ChunkMeta.chapter_title's docstring) — chapter_index is
# the clean, stable ordinal for filtering/keying, and is the only one
# exposed here.
_FILTERABLE_FIELDS = frozenset(
    {
        "source_id",
        "source_path",
        "source_title",
        "source_type",
        "chapter_index",
        "section",
        "page_start",
        "page_end",
        "chunk_index",
        "context_group_id",
        "token_count",
    }
)

_RANGE_OPS = {"gte": ">=", "gt": ">", "lte": "<=", "lt": "<", "eq": "=", "ne": "!="}


def _build_schema(dim: int) -> "pa.Schema":
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("source_id", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("source_title", pa.string()),
            pa.field("source_type", pa.string()),
            pa.field("section_path", pa.list_(pa.string())),
            pa.field("chapter_title", pa.string()),
            pa.field("chapter_index", pa.int64()),
            pa.field("section", pa.string()),
            pa.field("page_start", pa.int64()),
            pa.field("page_end", pa.int64()),
            pa.field("chunk_index", pa.int64()),
            pa.field("context_group_id", pa.string()),
            pa.field("char_start", pa.int64()),
            pa.field("char_end", pa.int64()),
            pa.field("token_count", pa.int64()),
            pa.field("extra_json", pa.string()),
        ]
    )


def _sql_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def _translate_filters(filters: Filter | None) -> str | None:
    if not filters:
        return None
    clauses: list[str] = []
    for key, value in filters.items():
        if key not in _FILTERABLE_FIELDS:
            raise StoreError(
                f"filter field {key!r} is not filterable; expected one of "
                f"{sorted(_FILTERABLE_FIELDS)}"
            )
        if isinstance(value, dict):
            for op, operand in value.items():
                sql_op = _RANGE_OPS.get(op)
                if sql_op is None:
                    raise StoreError(f"unsupported filter operator {op!r} for {key!r}")
                clauses.append(f"{key} {sql_op} {_sql_literal(operand)}")
        elif isinstance(value, (list, tuple, set)):
            literals = ", ".join(_sql_literal(v) for v in value)
            clauses.append(f"{key} IN ({literals})")
        else:
            clauses.append(f"{key} = {_sql_literal(value)}")
    return " AND ".join(clauses)


def _chunk_to_row(chunk: Chunk) -> dict:
    if chunk.vector is None:
        raise StoreError(f"chunk {chunk.chunk_id!r} has no vector; embed before upsert")
    m = chunk.meta
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "vector": chunk.vector,
        "source_id": m.source_id,
        "source_path": m.source_path,
        "source_title": m.source_title,
        "source_type": m.source_type,
        "section_path": list(m.section_path),
        "chapter_title": m.chapter_title,
        "chapter_index": m.chapter_index,
        "section": m.section,
        "page_start": m.page_start,
        "page_end": m.page_end,
        "chunk_index": m.chunk_index,
        "context_group_id": m.context_group_id,
        "char_start": m.char_start,
        "char_end": m.char_end,
        "token_count": m.token_count,
        "extra_json": json.dumps(m.extra),
    }


def _row_to_chunk(row: dict) -> Chunk:
    meta = ChunkMeta(
        source_id=row["source_id"],
        source_path=row["source_path"],
        source_title=row.get("source_title"),
        source_type=row["source_type"],
        section_path=tuple(row.get("section_path") or ()),
        chapter_title=row.get("chapter_title"),
        chapter_index=row.get("chapter_index"),
        section=row.get("section"),
        page_start=row.get("page_start"),
        page_end=row.get("page_end"),
        chunk_index=row["chunk_index"],
        context_group_id=row["context_group_id"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        token_count=row["token_count"],
        extra=json.loads(row["extra_json"]) if row.get("extra_json") else {},
    )
    return Chunk(chunk_id=row["chunk_id"], text=row["text"], meta=meta, vector=row.get("vector"))


class LanceDBStore:
    """Default VectorStore: an embedded, on-disk LanceDB table."""

    def __init__(
        self,
        path: str | Path,
        *,
        dim: int,
        table_name: str = "chunks",
        metric: str = "cosine",
    ) -> None:
        if lancedb is None:
            raise StoreError(
                "LanceDB storage requires the 'lancedb' extra: install ragcore[lancedb]"
            ) from _IMPORT_ERROR

        self.dim = dim
        self._metric = metric
        self._db = lancedb.connect(str(path))
        self._table_name = table_name

        if table_name in self._db.list_tables().tables:
            self._table = self._db.open_table(table_name)
        else:
            self._table = self._db.create_table(table_name, schema=_build_schema(dim))

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        for c in chunks:
            if c.meta.source_type == "synthetic":
                raise StoreError(
                    f"chunk {c.chunk_id!r} has source_type='synthetic' — synthetic "
                    "chunks (e.g. community summaries) are generated at retrieval "
                    "time and must never be upserted into a VectorStore"
                )
        rows = [_chunk_to_row(c) for c in chunks]
        ids = [r["chunk_id"] for r in rows]
        id_list = ", ".join(_sql_literal(i) for i in ids)
        self._table.delete(f"chunk_id IN ({id_list})")
        self._table.add(rows)

    def query(self, vector: Vector, k: int, filters: Filter | None = None) -> list[Retrieved]:
        q = self._table.search(vector).metric(self._metric).limit(k)
        where = _translate_filters(filters)
        if where:
            q = q.where(where)
        rows = q.to_list()

        results: list[Retrieved] = []
        for row in rows:
            distance = row.pop("_distance", None)
            # Monotonic transform of distance into a bounded, higher-is-better
            # score. Not a calibrated probability — just consistent ordering
            # across store backends that report distance rather than similarity.
            score = 1.0 / (1.0 + max(distance, 0.0)) if distance is not None else 0.0
            results.append(Retrieved(chunk=_row_to_chunk(row), score=score))
        return results

    def get(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        id_list = ", ".join(_sql_literal(i) for i in chunk_ids)
        rows = self._table.search().where(f"chunk_id IN ({id_list})").limit(len(chunk_ids)).to_list()
        return [_row_to_chunk(row) for row in rows]

    def get_by_source(self, source_id: str) -> list[Chunk]:
        rows = (
            self._table.search()
            .where(f"source_id = {_sql_literal(source_id)}")
            .limit(self.count())
            .to_list()
        )
        chunks = [_row_to_chunk(row) for row in rows]
        chunks.sort(key=lambda c: c.meta.chunk_index)
        return chunks

    def delete(
        self, *, source_id: str | None = None, chunk_ids: Sequence[str] | None = None
    ) -> int:
        if (source_id is None) == (not chunk_ids):
            raise StoreError("delete() requires exactly one of source_id or chunk_ids")

        predicate = (
            f"source_id = {_sql_literal(source_id)}"
            if source_id is not None
            else f"chunk_id IN ({', '.join(_sql_literal(i) for i in chunk_ids)})"
        )
        before = self.count()
        self._table.delete(predicate)
        after = self.count()
        return before - after

    def list_sources(self) -> list[SourceInfo]:
        # Pull only the columns needed for the summary, via pyarrow directly
        # (not to_pandas(), which would pull in pandas as an undeclared
        # dependency of the 'lancedb' extra).
        table = self._table.to_arrow().select(
            ["source_id", "source_path", "source_title", "source_type"]
        )
        counts: dict[str, int] = {}
        first_row: dict[str, dict] = {}
        for row in table.to_pylist():
            sid = row["source_id"]
            counts[sid] = counts.get(sid, 0) + 1
            first_row.setdefault(sid, row)

        return [
            SourceInfo(
                source_id=sid,
                path=first_row[sid]["source_path"],
                title=first_row[sid]["source_title"],
                source_type=first_row[sid]["source_type"],
                chunk_count=n,
            )
            for sid, n in counts.items()
        ]

    def count(self) -> int:
        return self._table.count_rows()

    # -- versioning (item 7: incremental ingestion) ---------------------

    def snapshot(self, tag: str) -> int:
        """Tags the table's current version for later rollback via
        restore_snapshot(tag) — uses LanceDB's own native version/tag
        support (confirmed present in the installed lancedb, previously
        unused by this store). A tagged version is exempt from
        `cleanup_old_versions()`/`optimize()`, so a consumer running those
        independently can't silently delete a snapshot out from under this
        store. Returns the tagged version number."""
        version = self._table.version
        self._table.tags.create(tag, version)
        return version

    def restore_snapshot(self, tag: str) -> int:
        """Rolls back to the version tagged `tag`. LanceDB's version history
        is append-only — `restore()` creates a NEW version equivalent to the
        tagged one rather than destroying anything since, so a rollback is
        itself re-runnable and never loses forward history. Returns the new
        current version number."""
        version = self._table.tags.get_version(tag)
        self._table.restore(version)
        return self._table.version

    def list_snapshots(self) -> list[tuple[str, int]]:
        """Every (tag, version) pair currently recorded, for a caller
        implementing retention (see ragcore.versioning.snapshot)."""
        return [(tag, info["version"]) for tag, info in self._table.tags.list().items()]

    def drop_snapshot(self, tag: str) -> None:
        """Removes a tag. The tagged version itself is not deleted by this
        call — dropping the tag only re-exposes that version to normal
        cleanup (`cleanup_old_versions()`/`optimize()`), which is the
        intended garbage-collection path."""
        self._table.tags.delete(tag)
