"""Requirement 4: query load — hundreds of queries against the combined
index, latency measured and asserted on, hybrid retrieval (vector + graph)
verified correct at library scale, not just the 7-chapter case.

Includes the direct regression test for the GraphRetriever dead-code defect
(every expansion scored below every seed, so `merged[:k]` truncated all of
them away whenever vector search alone returned >= k results — invisible
in the original unit tests, which all used exactly one seed). See
docs/INTEGRATION_TEST_FINDINGS.md.
"""

from __future__ import annotations

import json
import string
import time
from pathlib import Path

from ragcore.retrieval.graph import GraphRetriever
from tests.integration.conftest import SharedLibrary
from tests.integration.eval.queryset import needle_queryset
from tests.integration.eval.report import render_table, run_eval
from tests.integration.scale import is_quick
from tests.integration.synth.generator import THEMES, generate_paragraph
from tests.integration.synth.library import find_chunks_with_marker

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def _generated_queries(n: int) -> list[str]:
    """Synthetic query load beyond the needle set: plausible-shaped
    sentences drawn from the same themed vocabularies as the corpus, so
    they exercise real (if noisy) vector search rather than being
    pathological inputs — those are covered separately below and in
    test_failure_injection.py."""
    import random

    rng = random.Random(4242)
    themes = list(THEMES.values())
    out = []
    for _ in range(n):
        theme = rng.choice(themes)
        out.append(generate_paragraph(rng, theme, n_sentences=1))
    return out


def test_hundreds_of_queries_run_without_error_and_stay_fast(shared_library: SharedLibrary) -> None:
    n = 60 if is_quick() else 400
    queries = _generated_queries(n)
    corpus = shared_library.corpus

    latencies_ms: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        results = corpus.retrieve(q, k=10)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        assert isinstance(results, list)
        assert len(results) <= 10
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "results not ordered by score"

    latencies_ms.sort()
    p50 = latencies_ms[len(latencies_ms) // 2]
    p95 = latencies_ms[int(len(latencies_ms) * 0.95)]
    p99 = latencies_ms[min(len(latencies_ms) - 1, int(len(latencies_ms) * 0.99))]

    # Generous absolute ceilings (this is a hashed bag-of-words embedder and
    # an in-process LanceDB table — not a network call) plus a shape check:
    # p99 blowing out relative to p50 catches an algorithmic regression
    # (e.g. an accidental O(n^2) scan) that a loose absolute ceiling alone
    # would miss, without being fragile to a slower CI box.
    assert p50 < 200, f"p50 latency regressed: {p50:.1f}ms"
    assert p99 < 2000, f"p99 latency regressed: {p99:.1f}ms"
    assert p99 / max(p50, 0.01) < 50, f"p99/p50 ratio blew out: {p99:.1f}ms / {p50:.1f}ms"

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "query_load_latency.json").write_text(
        json.dumps({"n_queries": n, "p50_ms": p50, "p95_ms": p95, "p99_ms": p99}, indent=2)
    )


def test_filters_k_and_min_score_honored_at_library_scale(shared_library: SharedLibrary) -> None:
    corpus = shared_library.corpus
    queries = _generated_queries(20 if is_quick() else 60)

    for q in queries:
        for k in (1, 5, 25):
            results = corpus.retrieve(q, k=k)
            assert len(results) <= k

        results = corpus.retrieve(q, k=10, min_score=0.99)  # unreasonably high bar
        assert all(r.score >= 0.99 for r in results)

    # source_id filter: every result must actually belong to the filtered source
    one_source = corpus.sources()[0]
    filtered = corpus.retrieve(queries[0], k=10, filters={"source_id": one_source.source_id})
    assert all(r.chunk.meta.source_id == one_source.source_id for r in filtered)


def test_needle_queries_reach_perfect_recall_at_library_scale(shared_library: SharedLibrary) -> None:
    """Sanity floor for the eval harness itself (full metrics live in
    test_eval_quality.py): every planted needle must still be findable once
    it's competing against every other book in the shared index, not just
    within its own book."""
    queries = needle_queryset(shared_library.corpus, shared_library.library.needles)
    assert len(queries) >= 1
    summary = run_eval(
        lambda q, k: shared_library.corpus.retrieve(q, k=k), queries, k=10, label="needles@library-scale"
    )
    assert summary.recall_at_k == 1.0, render_table([summary])


def test_graph_hybrid_reaches_content_pure_vector_search_misses(shared_library: SharedLibrary) -> None:
    """Direct regression test for the GraphRetriever k-truncation defect.
    See tests/integration/synth/library.py's GraphProbe for how the
    scenario is planted: a "commands" relation between Captain Aurelia Venn
    and Obadiah stated in one chapter, Obadiah mentioned again — with no
    lexical overlap with the query at all — in a distant chapter. At
    library scale (many candidate chunks), vector search alone should NOT
    surface the distant chunk; graph-hybrid retrieval, with the fixed
    k-reservation, should.
    """
    if not shared_library.is_ingested("clean_toc"):
        import pytest

        pytest.skip("needs clean_toc ingested (RAGCORE_IT_SCALE=quick subset)")

    probe = shared_library.book("clean_toc").graph_probe
    assert probe is not None
    corpus = shared_library.corpus
    target_chunks = set(find_chunks_with_marker(corpus, probe.target_marker))
    assert target_chunks, "graph probe's target chunk not found — planting broke"

    vector_only_ids = {r.chunk.chunk_id for r in corpus.retrieve(probe.seed_query, k=10)}
    assert not (vector_only_ids & target_chunks), (
        "test premise broken: vector-only search already finds the target chunk "
        "(the scenario needs to require graph traversal to be meaningful)"
    )

    hybrid = GraphRetriever(corpus.retriever, shared_library.graph, corpus)
    hybrid_results = hybrid.retrieve(probe.seed_query, k=10)
    hybrid_ids = {r.chunk.chunk_id for r in hybrid_results}
    assert hybrid_ids & target_chunks, "graph-hybrid retrieval failed to surface graph-reachable content"
    assert len(hybrid_results) <= 10  # k contract still holds

    graph_hit = next(r for r in hybrid_results if r.chunk.chunk_id in target_chunks)
    assert graph_hit.graph is not None
    seed_scores = [r.score for r in hybrid_results if r.graph is None]
    assert seed_scores, "expected at least one vector seed alongside the graph hit"
    assert graph_hit.score < min(seed_scores), "graph hit should rank below every vector seed"


def test_graph_hybrid_never_exceeds_k_and_never_drops_below_available_seeds(
    shared_library: SharedLibrary,
) -> None:
    """The k-reservation fix must not regress the ordinary "no graph signal
    for this query" case: a query with nothing graph-adjacent should still
    get a full k vector-only results (the reserved slice gets backfilled
    from the same single inner call), not silently fewer."""
    corpus = shared_library.corpus
    hybrid = GraphRetriever(corpus.retriever, shared_library.graph, corpus)

    for q in _generated_queries(10):
        for k in (1, 5, 10, 20):
            results = hybrid.retrieve(q, k=k)
            assert len(results) <= k
            vector_only = corpus.retrieve(q, k=k)
            # never fewer results than plain vector search would return
            # for the same k, purely because a graph reservation went unused
            assert len(results) >= len(vector_only) or len(vector_only) < k


def test_unicode_and_punctuation_queries_do_not_crash(shared_library: SharedLibrary) -> None:
    corpus = shared_library.corpus
    odd_queries = [
        "",
        " ",
        "?" * 50,
        "café résumé naïve",
        "éèê " * 10,
        "".join(string.punctuation),
        "a" * 5000,
        "SELECT * FROM chunks; -- ",
        "北京 東京 москва",
        "\n\t\r  weird\x00control",
    ]
    for q in odd_queries:
        results = corpus.retrieve(q, k=5)
        assert isinstance(results, list)
        assert len(results) <= 5
