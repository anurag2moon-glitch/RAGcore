# LanceDBStore benchmarks

Captured with `scripts/benchmark_lancedb.py` (Phase 2 gap list item 11:
"benchmarks across corpus sizes"). Re-run it yourself before relying on
a specific number for capacity planning — these numbers are from one
development machine (macOS, arm64, Python 3.12.10, lancedb 0.37.1), not
a controlled benchmarking environment, and are the largest sizes
practical to run in that environment in reasonable time. 1M/10M-chunk
runs are left to be executed on real target hardware; the script is
written to run correctly at those sizes (it was validated for
correctness up to 50,000, not performance-tuned beyond it).

**Read the recall@10 column with the caveat below before drawing any
conclusion from it.**

## Results: dim=1024 (voyage-context-4's default output dimension), k=10, 30 queries, method=ivf_pq

| Chunks | Upsert (chunks/sec) | Query p50, no index (ms) | Query p50, indexed (ms) | Index build (s) | recall@10 (see caveat) |
|---|---|---|---|---|---|
| 1,000 | 4,593 | 2.48 | 2.36 | 0.22 | 0.310 |
| 10,000 | 8,591 | 8.81 | 3.76 | 4.78 | 0.103 |
| 50,000 | 8,399 | 30.74 | 4.72 | 39.47 | 0.047 |

## What this shows

- **Upsert throughput is roughly flat** (~4.6k-8.6k chunks/sec) across
  this range — no evidence of upsert degrading as the table grows, at
  least up to 50k rows on this machine.
- **Brute-force query latency grows with corpus size**, as expected for
  an exact scan (2.5ms → 8.8ms → 30.7ms p50 as rows go 1k → 10k → 50k).
  **Indexed query latency stays far flatter** (2.4ms → 3.8ms → 4.7ms
  p50 over the same range) — the entire point of building a vector
  index, and the crossover is already clearly visible by 10k rows on
  this machine: `create_vector_index()` is worth its one-time build cost
  once a corpus is more than a few thousand chunks and query latency
  matters.
- **Index build time grows faster than linearly in this range** (0.22s →
  4.78s → 39.47s for 1k → 10k → 50k) in this configuration
  (`num_partitions` capped at `min(256, size // 100)` by the benchmark
  script, not a `LanceDBStore` default) — budget for this as a real,
  non-trivial one-time (or per-`compact()`) cost at larger scale, not
  something to call on every ingest.

## The recall@10 caveat

The recall numbers above (0.31 → 0.10 → 0.05, declining as corpus size
grows) are **not a realistic estimate of recall on a real corpus**.
Query and corpus vectors in this benchmark are uniform random noise with
no semantic structure — the worst possible input for IVF-PQ/HNSW, which
both rely on real data clustering to partition the space usefully. Real
text embeddings cluster by meaning; random noise in a 1024-dimensional
space is close to uniformly distant from everything (the curse of
dimensionality), which is exactly the regime where an IVF partitioning
scheme degrades hardest. The declining trend with size is itself an
artifact of this: more random points means less separation between
"true" and "false" neighbors for an approximate method to exploit.

What this benchmark *does* faithfully measure, independent of the data
being random, is throughput and latency, and that
`create_vector_index()`/`compact()`/`index_stats()` work correctly end
to end (also covered directly by
`tests/test_lancedb_index_lifecycle.py`, which asserts exact nearest-
neighbor correctness on a small, controlled table). Measure recall
against your own embedder's real output on your own corpus — e.g. via
`ragcore.eval.run_eval()` with `ClaudeVerifier` — before setting a
production recall threshold.
