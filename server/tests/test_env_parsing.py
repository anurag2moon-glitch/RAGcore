"""app.env_parsing: parse_colon_pairs / parse_int_quotas — the shared
parser three different provider modules (auth, quota, extraction_quota)
used to each duplicate independently.
"""

from __future__ import annotations

import pytest
from app.env_parsing import parse_colon_pairs, parse_int_quotas


def test_parse_colon_pairs() -> None:
    assert parse_colon_pairs("a:1,b:2", env_var="X") == {"a": "1", "b": "2"}


def test_parse_colon_pairs_ignores_blank_entries() -> None:
    assert parse_colon_pairs("a:1,,b:2,", env_var="X") == {"a": "1", "b": "2"}


def test_parse_colon_pairs_rejects_a_malformed_entry() -> None:
    with pytest.raises(RuntimeError, match="X entry 'no-colon' is not in 'left:right' form"):
        parse_colon_pairs("no-colon", env_var="X")


def test_parse_colon_pairs_rejects_a_missing_right_side() -> None:
    with pytest.raises(RuntimeError, match="not in 'left:right' form"):
        parse_colon_pairs("a:", env_var="X")


def test_parse_int_quotas() -> None:
    assert parse_int_quotas("tenant-a:1000,tenant-b:5000", env_var="X") == {
        "tenant-a": 1000,
        "tenant-b": 5000,
    }


def test_parse_int_quotas_rejects_a_non_integer_limit() -> None:
    with pytest.raises(RuntimeError, match="non-integer limit"):
        parse_int_quotas("tenant-a:not-a-number", env_var="X")


def test_parse_int_quotas_rejects_a_malformed_entry() -> None:
    with pytest.raises(RuntimeError, match="not in 'left:right' form"):
        parse_int_quotas("no-colon", env_var="X")
