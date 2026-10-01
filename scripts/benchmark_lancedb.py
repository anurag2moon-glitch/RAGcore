"""Throughput/latency benchmark for LanceDBStore across corpus sizes —
Phase 2's "benchmarks across corpus sizes (10k, 1M, 10M chunks)" gap list
item. Not part of the library's public API — a convenience script,
matching scripts/query.py's shape.

No network, no API keys: chunks are synthetic (random vectors, minimal
valid ChunkMeta), so this measures LanceDBStore/LanceDB itself, not an
embedder. For each requested size it measures, in order:

  1. upsert throughput (chunks/sec), batched
  2. query latency (p50/p95/p99, ms) with no vector index — brute-force
  3. create_vector_index() build time
  4. query latency again, now indexed — compare against step 2
  5. recall@10: how many of the indexed top-10 match the brute-force
     top-10 (ground truth), averaged over the same query set — the real
     quality cost IVF-PQ's approximation trades for its speed, not just a
     claim that it's "approximate"

Usage:
    python scripts/benchmark_lancedb.py                      # default sizes: 1000,10000
    python scripts/benchmark_lancedb.py --sizes 1000,10000,100000
    python scripts/benchmark_lancedb.py --sizes 1000000 --queries 20  # fewer queries at huge scale
    python scripts/benchmark_lancedb.py --dim 1536 --method hnsw_sq

A 1M or 10M run takes real time and disk (expect single-digit minutes to
tens of minutes for 1M depending on hardware, much more for 10M) — this
script is written to run correctly at those sizes, but this repository's
own checked-in numbers (see docs/BENCHMARKS.md) were captured at smaller
sizes only; run it yourself at the scale you actually care about before
trusting a specific number for capacity planning.

**The recall@k number is a worst case, not a representative one.** Query
and corpus vectors here are uniform random noise with no real semantic
structure — exactly the input IVF-PQ's partitioning handles worst,
since approximate search relies on real clustering to be both fast and
accurate. Expect meaningfully higher recall on real embeddings (which
cluster by meaning) than this script reports; what this script DOES
faithfully measure is throughput/latency (not data-dependent) and that
the create_vector_index()/compact() mechanism itself works end to end.
Measure recall against your own embedder's real output, on your own
corpus, before trusting a recall number for a production threshold.
"""

from __future__ import annotations

import argparse
import random
import statistics
import tempfile
import time
from pathlib import Path

from ragcore.stores.lancedb import LanceDBStore
from ragcore.types import Chunk, ChunkMeta

UPSERT_BATCH_SIZE = 1000


def _synthetic_chunk(i: int, dim: int, rng: random.Random) -> Chunk:
    meta = ChunkMeta(
        source_id=f"source-{i // 500}",  # ~500 chunks/book, for realistic source_id cardinality
        source_path=f"book-{i // 500}.md",
        source_title=None,
        source_type="md",
        section_path=(),
        chapter_title=None,
        chapter_index=None,
        section=None,
        page_start=None,
        page_end=None,
        chunk_index=i,
        context_group_id=f"group-{i // 50}",
        char_start=0,
        char_end=1,
        token_count=1,
    )
    vector = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    return Chunk(chunk_id=f"c{i}", text=f"synthetic chunk {i}", meta=meta, vector=vector)


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    s = sorted(samples_ms)
    n = len(s)
    return {
        "p50_ms": s[int(n * 0.50)],
        "p95_ms": s[min(n - 1, int(n * 0.95))],
        "p99_ms": s[min(n - 1, int(n * 0.99))],
    }


def _time_queries(store: LanceDBStore, queries: list[list[float]], k: int) -> list[float]:
    samples_ms: list[float] = []
    for q in queries:
        start = time.perf_counter()
        store.query(q, k=k)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return samples_ms


def run_one_size(
    *, size: int, dim: int, k: int, num_queries: int, method: str, seed: int
) -> None:
    rng = random.Random(seed)
    print(f"\n=== size={size:,} dim={dim} method={method} ===")

    with tempfile.TemporaryDirectory() as d:
        store = LanceDBStore(Path(d) / "bench.lancedb", dim=dim)

        t0 = time.perf_counter()
        batch: list[Chunk] = []
        for i in range(size):
            batch.append(_synthetic_chunk(i, dim, rng))
            if len(batch) >= UPSERT_BATCH_SIZE:
                store.upsert(batch)
                batch = []
        if batch:
            store.upsert(batch)
        upsert_elapsed = time.perf_counter() - t0
        print(f"upsert: {size / upsert_elapsed:,.0f} chunks/sec ({upsert_elapsed:.2f}s total)")

        query_vectors = [[rng.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(num_queries)]

        brute_force_ms = _time_queries(store, query_vectors, k)
        bf_pct = _percentiles(brute_force_ms)
        print(
            f"query (brute-force, no index): p50={bf_pct['p50_ms']:.2f}ms "
            f"p95={bf_pct['p95_ms']:.2f}ms p99={bf_pct['p99_ms']:.2f}ms"
        )
        # ground truth for recall, from the same (unindexed) store
        ground_truth = [
            {r.chunk.chunk_id for r in store.query(q, k=k)} for q in query_vectors
        ]

        num_partitions = max(1, min(256, size // 100)) or None
        t0 = time.perf_counter()
        store.create_vector_index(method=method, num_partitions=num_partitions)
        index_build_s = time.perf_counter() - t0
        print(f"create_vector_index({method}): {index_build_s:.2f}s")

        indexed_ms = _time_queries(store, query_vectors, k)
        idx_pct = _percentiles(indexed_ms)
        print(
            f"query (indexed): p50={idx_pct['p50_ms']:.2f}ms "
            f"p95={idx_pct['p95_ms']:.2f}ms p99={idx_pct['p99_ms']:.2f}ms"
        )

        indexed_results = [
            {r.chunk.chunk_id for r in store.query(q, k=k)} for q in query_vectors
        ]
        recall = statistics.mean(
            len(truth & got) / len(truth) if truth else 1.0
            for truth, got in zip(ground_truth, indexed_results, strict=True)
        )
        print(f"recall@{k} (indexed vs. brute-force ground truth): {recall:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default="1000,10000", help="comma-separated chunk counts")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=50, dest="num_queries")
    parser.add_argument("--method", choices=["ivf_pq", "hnsw_sq"], default="ivf_pq")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    print(
        "NOTE: recall@k below is measured on structureless random vectors "
        "(a worst case for IVF-PQ/HNSW) — see this script's module "
        "docstring. Throughput/latency numbers are representative; the "
        "recall number is not."
    )
    for size in sizes:
        run_one_size(
            size=size,
            dim=args.dim,
            k=args.k,
            num_queries=args.num_queries,
            method=args.method,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
