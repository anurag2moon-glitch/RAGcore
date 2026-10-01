"""Unit tests for SqliteEmbeddingCache in isolation from Corpus."""

from __future__ import annotations

from pathlib import Path

from ragcore.embedding.cache import SqliteEmbeddingCache


def test_get_many_on_empty_cache_returns_nothing(tmp_path: Path) -> None:
    cache = SqliteEmbeddingCache(tmp_path / "cache.sqlite3")
    assert cache.get_many([("h1", "model-a", 8)]) == {}
    assert cache.get_many([]) == {}


def test_put_then_get_round_trips_the_vector(tmp_path: Path) -> None:
    cache = SqliteEmbeddingCache(tmp_path / "cache.sqlite3")
    key = ("h1", "model-a", 4)
    cache.put_many({key: [1.0, 2.5, -3.25, 0.0]})

    result = cache.get_many([key])
    assert result[key] == [1.0, 2.5, -3.25, 0.0]


def test_different_model_or_dim_is_a_different_key(tmp_path: Path) -> None:
    cache = SqliteEmbeddingCache(tmp_path / "cache.sqlite3")
    cache.put_many({("h1", "model-a", 4): [1.0, 2.0, 3.0, 4.0]})

    assert cache.get_many([("h1", "model-b", 4)]) == {}  # different model
    assert cache.get_many([("h1", "model-a", 8)]) == {}  # different dim


def test_put_many_overwrites_an_existing_key(tmp_path: Path) -> None:
    cache = SqliteEmbeddingCache(tmp_path / "cache.sqlite3")
    key = ("h1", "model-a", 2)
    cache.put_many({key: [1.0, 2.0]})
    cache.put_many({key: [9.0, 9.0]})

    assert cache.get_many([key])[key] == [9.0, 9.0]
    assert cache.count() == 1


def test_get_many_handles_more_than_one_sqlite_batch(tmp_path: Path) -> None:
    cache = SqliteEmbeddingCache(tmp_path / "cache.sqlite3")
    items = {(f"h{i}", "model-a", 2): [float(i), float(i)] for i in range(1200)}
    cache.put_many(items)

    result = cache.get_many(list(items.keys()))
    assert len(result) == 1200
    assert result[("h999", "model-a", 2)] == [999.0, 999.0]


def test_float32_round_trip_preserves_reasonable_precision(tmp_path: Path) -> None:
    cache = SqliteEmbeddingCache(tmp_path / "cache.sqlite3")
    key = ("h1", "model-a", 1)
    original = [0.123456789]
    cache.put_many({key: original})

    [restored] = cache.get_many([key])[key]
    assert abs(restored - original[0]) < 1e-6
