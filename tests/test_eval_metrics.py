"""Pure metric functions in ragcore.eval.metrics — no ragcore imports, no
corpus, just the math."""

from __future__ import annotations

from ragcore.eval.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_counts_gold_hits_within_the_top_k() -> None:
    ranked = ["a", "b", "c", "d"]
    relevant = {"b", "d", "z"}
    assert recall_at_k(ranked, relevant, k=2) == 1 / 3  # only "b" is in top 2
    assert recall_at_k(ranked, relevant, k=4) == 2 / 3


def test_recall_at_k_on_empty_relevant_is_zero_not_an_error() -> None:
    assert recall_at_k(["a"], set(), k=1) == 0.0


def test_precision_at_k() -> None:
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, k=3) == 2 / 3
    assert precision_at_k(["a", "b"], set(), k=2) == 0.0


def test_hit_rate_at_k_is_binary() -> None:
    assert hit_rate_at_k(["a", "b"], {"b"}, k=2) == 1.0
    assert hit_rate_at_k(["a", "b"], {"z"}, k=2) == 0.0
    assert hit_rate_at_k(["a"], set(), k=1) == 0.0


def test_ndcg_at_k_rewards_earlier_ranking_of_relevant_items() -> None:
    relevant = {"a"}
    early = ndcg_at_k(["a", "b", "c"], relevant, k=3)
    late = ndcg_at_k(["b", "c", "a"], relevant, k=3)
    assert early == 1.0  # perfect ranking
    assert 0.0 < late < early


def test_ndcg_at_k_on_empty_relevant_is_zero() -> None:
    assert ndcg_at_k(["a"], set(), k=1) == 0.0


def test_reciprocal_rank_finds_the_first_relevant_hit() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0
    assert reciprocal_rank(["a"], set()) == 0.0
