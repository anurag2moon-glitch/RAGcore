"""Requirement 6: full re-run of the eval harness (Recall@k, nDCG@10, MRR)
at library scale, to catch any quality regression from a smaller-scale
validation.

There was no eval harness anywhere in ragcore before this suite (see
tests/integration/eval/metrics.py's docstring) — "re-run at larger scale"
first meant building it, then running it at two genuinely different scales
with the *same* metric code so the comparison means something. Rather than
depending on the gitignored, possibly-absent samples/library.lancedb
7-chapter fixture (see tests/test_graph_fixture_pride_prejudice.py), the
"smaller scale" here is a 3-book subset of this suite's own hermetic
synthetic library, and "larger scale" is the full 8-book shared_library —
same needle-based ground truth methodology, same metric functions, run
against the identical query set at both scales so any degradation is
attributable to scale alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragcore.corpus import Corpus
from ragcore.retrieval.graph import GraphRetriever
from tests.integration.conftest import SharedLibrary
from tests.integration.eval.queryset import needle_queryset
from tests.integration.eval.report import render_table, run_eval
from tests.integration.scale import is_quick
from tests.integration.synth.library import Library

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
BASELINE_PATH = Path(__file__).parent / "baselines" / "eval_baseline.json"

SMALL_SCALE_BOOK_KEYS = ("clean_toc", "fragment_anchored", "unusual_naming")

# Regression gate tolerances — generous on purpose: this eval harness
# exists to catch a real pipeline regression (a broken filter, a k-cutoff
# bug, a resolver change that quietly drops recall), not to chase a
# specific float against a lexical mock embedder's incidental behavior.
MIN_RECALL_AT_K = 0.90
MIN_NDCG_AT_K = 0.80
MIN_MRR = 0.80
MAX_P95_LATENCY_MS = 500.0


def test_needle_eval_at_small_scale(fresh_corpus: Corpus, synth_library: Library) -> None:
    for key in SMALL_SCALE_BOOK_KEYS:
        fresh_corpus.ingest(synth_library.book(key).path)

    needles = [n for n in synth_library.needles if n.book_key in SMALL_SCALE_BOOK_KEYS]
    queries = needle_queryset(fresh_corpus, needles)
    assert len(queries) == len(SMALL_SCALE_BOOK_KEYS)

    summary = run_eval(lambda q, k: fresh_corpus.retrieve(q, k=k), queries, k=10, label="small-scale")
    assert summary.recall_at_k >= MIN_RECALL_AT_K, render_table([summary])
    assert summary.ndcg_at_k >= MIN_NDCG_AT_K, render_table([summary])
    assert summary.mrr >= MIN_MRR, render_table([summary])


def test_needle_eval_at_full_library_scale_no_regression_vs_small_scale(
    shared_library: SharedLibrary,
) -> None:
    """The actual "re-run at larger scale" check: the SAME needle queries
    used at small scale must still resolve correctly once competing
    against every other book's chunks in one shared index — this is
    exactly the kind of regression (a query's true answer gets crowded out
    by unrelated but lexically-similar content once the corpus grows) that
    a 7-chapter-only validation could never have caught.
    """
    needles = [n for n in shared_library.library.needles if n.book_key in SMALL_SCALE_BOOK_KEYS]
    queries = needle_queryset(shared_library.corpus, needles)
    if len(queries) < len(SMALL_SCALE_BOOK_KEYS):
        pytest.skip("small-scale comparison books not fully ingested (RAGCORE_IT_SCALE=quick)")

    summary = run_eval(
        lambda q, k: shared_library.corpus.retrieve(q, k=k), queries, k=10, label="same-queries@full-scale"
    )
    assert summary.recall_at_k >= MIN_RECALL_AT_K, (
        f"recall regressed at library scale vs small scale:\n{render_table([summary])}"
    )
    assert summary.ndcg_at_k >= MIN_NDCG_AT_K, render_table([summary])
    assert summary.mrr >= MIN_MRR, render_table([summary])


def test_full_eval_report_vector_only_vs_graph_hybrid(shared_library: SharedLibrary) -> None:
    """The full harness run this requirement asks for: every planted
    needle in the (possibly full) shared library, vector-only vs
    graph-hybrid, written to a JSON artifact and gated against the
    committed baseline."""
    queries = needle_queryset(shared_library.corpus, shared_library.library.needles)
    assert queries, "no needle queries resolved — ingestion or planting broke"

    vector_summary = run_eval(
        lambda q, k: shared_library.corpus.retrieve(q, k=k), queries, k=10, label="vector-only"
    )
    hybrid = GraphRetriever(shared_library.corpus.retriever, shared_library.graph, shared_library.corpus)
    hybrid_summary = run_eval(lambda q, k: hybrid.retrieve(q, k=k), queries, k=10, label="graph-hybrid")

    table = render_table([vector_summary, hybrid_summary])

    assert vector_summary.recall_at_k >= MIN_RECALL_AT_K, table
    assert vector_summary.ndcg_at_k >= MIN_NDCG_AT_K, table
    assert vector_summary.mrr >= MIN_MRR, table
    assert vector_summary.latency_p95_ms <= MAX_P95_LATENCY_MS, table
    # Graph-hybrid must never regress recall below vector-only: it should
    # only ever add candidates, never suppress a vector hit (see
    # GraphRetriever's module docstring).
    assert hybrid_summary.recall_at_k >= vector_summary.recall_at_k - 1e-9, table

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    result = {
        "book_count": len(shared_library.book_keys),
        "n_needles": len(queries),
        "vector_only": vector_summary.to_dict(),
        "graph_hybrid": hybrid_summary.to_dict(),
    }
    (ARTIFACTS_DIR / "eval_report.json").write_text(json.dumps(result, indent=2))

    if BASELINE_PATH.exists() and not is_quick():
        baseline = json.loads(BASELINE_PATH.read_text())
        base_recall = baseline["vector_only"]["recall_at_k"]
        assert vector_summary.recall_at_k >= base_recall - 0.05, (
            f"recall@k regressed vs committed baseline: {vector_summary.recall_at_k:.3f} "
            f"< baseline {base_recall:.3f} - 0.05\n{table}"
        )
