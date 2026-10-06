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
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, Sequence

from ragcore.errors import StoreError, wrap_backend_errors
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
        "tenant_id",
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
            pa.field("tenant_id", pa.string()),
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
        "tenant_id": m.tenant_id,
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
        tenant_id=row.get("tenant_id"),
    )
    return Chunk(chunk_id=row["chunk_id"], text=row["text"], meta=meta, vector=row.get("vector"))


class LanceDBStore:
    """Default VectorStore: an embedded, on-disk LanceDB table."""

    @wrap_backend_errors
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
            # Schema migration for a table created before tenant_id
            # existed: add_columns() backfills every existing row with
            # NULL and leaves everything else untouched — confirmed
            # directly (not assumed), since LanceDB raises a ValueError
            # on add()/upsert() the moment a row carries a key absent
            # from an already-open table's schema, which this avoids.
            if "tenant_id" not in self._table.schema.names:
                self._table.add_columns({"tenant_id": "CAST(NULL AS STRING)"})
        else:
            self._table = self._db.create_table(table_name, schema=_build_schema(dim))

    @wrap_backend_errors
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

    @wrap_backend_errors
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

    @wrap_backend_errors
    def get(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        id_list = ", ".join(_sql_literal(i) for i in chunk_ids)
        rows = self._table.search().where(f"chunk_id IN ({id_list})").limit(len(chunk_ids)).to_list()
        return [_row_to_chunk(row) for row in rows]

    @wrap_backend_errors
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

    @wrap_backend_errors
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

    @wrap_backend_errors
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

    @wrap_backend_errors
    def count(self) -> int:
        return self._table.count_rows()

    # -- versioning (item 7: incremental ingestion) ---------------------

    @wrap_backend_errors
    def snapshot(self, tag: str) -> int:
        """Tags the table's current version for later rollback via
        restore_snapshot(tag) — uses LanceDB's own native version/tag
        support (confirmed present in the installed lancedb, previously
        unused by this store). A tagged version is *protected*, not
        silently exempted: `compact()` (LanceDB's `Table.optimize()`)
        raises a StoreError instead of proceeding if the requested cleanup
        window would remove a tagged version — confirmed empirically
        (`tests/test_lancedb_index_lifecycle.py::test_compact_refuses_to_remove_a_tagged_snapshot`),
        not merely assumed from LanceDB's docs. Drop the tag first
        (`drop_snapshot`) if you really want to reclaim that space.
        Returns the tagged version number."""
        version = self._table.version
        self._table.tags.create(tag, version)
        return version

    @wrap_backend_errors
    def restore_snapshot(self, tag: str) -> int:
        """Rolls back to the version tagged `tag`. LanceDB's version history
        is append-only — `restore()` creates a NEW version equivalent to the
        tagged one rather than destroying anything since, so a rollback is
        itself re-runnable and never loses forward history. Returns the new
        current version number."""
        version = self._table.tags.get_version(tag)
        self._table.restore(version)
        return self._table.version

    @wrap_backend_errors
    def list_snapshots(self) -> list[tuple[str, int]]:
        """Every (tag, version) pair currently recorded, for a caller
        implementing retention (see ragcore.versioning.snapshot)."""
        return [(tag, info["version"]) for tag, info in self._table.tags.list().items()]

    @wrap_backend_errors
    def drop_snapshot(self, tag: str) -> None:
        """Removes a tag. The tagged version itself is not deleted by this
        call — dropping the tag only re-exposes that version to normal
        cleanup (`cleanup_old_versions()`/`optimize()`), which is the
        intended garbage-collection path."""
        self._table.tags.delete(tag)

    # -- index lifecycle (item 11: compaction/tuning) ---------------------

    @wrap_backend_errors
    def create_vector_index(
        self,
        *,
        method: Literal["ivf_pq", "hnsw_sq"] = "ivf_pq",
        num_partitions: int | None = None,
        replace: bool = True,
    ) -> None:
        """Builds an approximate-nearest-neighbor index on the vector
        column. LanceDB does exact brute-force search until an index
        exists — fine at small/medium scale, which is why this isn't
        called automatically from `__init__`/`upsert`: an index build has
        a real one-time cost, and a small corpus gets no query-speed
        benefit from one, only that cost. Call this once a corpus is large
        enough that query() latency matters (LanceDB's own guidance:
        roughly tens of thousands of rows and up — see
        scripts/benchmark_lancedb.py for measured numbers at a few
        concrete sizes).

        `ivf_pq` (default): IVF-PQ, LanceDB's standard disk-efficient
        index. `hnsw_sq`: HNSW with scalar quantization — typically faster
        queries at a higher build cost and memory footprint; prefer it
        when query latency matters more than build time.

        `num_partitions=None` lets LanceDB choose a partition count from
        the table's current row count. Re-create the index (the default
        `replace=True`) after the corpus has grown substantially — a
        partition count sized for 10k rows under-partitions a table
        that's since grown to 1M, degrading recall.
        """
        from lancedb.index import HnswSq, IvfPq

        config = (
            IvfPq(distance_type=self._metric, num_partitions=num_partitions)
            if method == "ivf_pq"
            else HnswSq(distance_type=self._metric, num_partitions=num_partitions)
        )
        self._table.create_index("vector", config=config, replace=replace)

    @wrap_backend_errors
    def compact(self, *, cleanup_older_than_days: float | None = None) -> None:
        """Merges small write fragments into larger ones, prunes old table
        versions, and refreshes any existing vector index with data added
        since it was last built or compacted — LanceDB's three-in-one
        `Table.optimize()` (its own docstring: "Modeled after VACUUM in
        PostgreSQL"). If the requested cleanup window would remove a
        tagged snapshot (see `snapshot()`), this raises a `StoreError`
        instead of silently skipping or silently deleting it — compaction
        never invalidates a snapshot out from under a caller relying on
        it; drop the tag first (`drop_snapshot`) if that space really
        should be reclaimed.

        `cleanup_older_than_days=None` keeps LanceDB's own default
        (versions older than 7 days are pruned); pass `0` to prune every
        non-latest version immediately (subject to the tagged-version
        protection above).
        """
        cleanup = (
            timedelta(days=cleanup_older_than_days)
            if cleanup_older_than_days is not None
            else None
        )
        self._table.optimize(cleanup_older_than=cleanup)

    @wrap_backend_errors
    def index_stats(self) -> list[dict[str, Any]]:
        """One summary dict per index currently on the table — lets a
        caller check whether `create_vector_index()` has actually taken
        effect, or whether rows added after the index was built are still
        falling back to a brute-force scan for the unindexed portion
        (`num_unindexed_rows` — nonzero means `compact()` or a fresh
        `create_vector_index()` call would fold them in)."""
        return [
            {
                "name": cfg.name,
                "columns": list(cfg.columns),
                "index_type": cfg.index_type,
                "num_indexed_rows": cfg.num_indexed_rows,
                "num_unindexed_rows": cfg.num_unindexed_rows,
            }
            for cfg in self._table.list_indices()
        ]
