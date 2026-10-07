"""Shared parsers for this server's "comma-separated key:value pairs"
environment variable convention (`RAGCORE_API_KEYS`,
`RAGCORE_TENANT_QUOTAS`, `RAGCORE_EXTRACTION_TOKEN_QUOTAS`) — written
once here and imported everywhere, rather than the three
near-identical copies that existed before `app.settings` needed one
place to validate every env var eagerly at startup. `RuntimeError` on
a malformed entry, same as before this module existed — callers that
want a different error type (`app.settings`'s own `pydantic`
validation, specifically) catch and re-raise.
"""

from __future__ import annotations


def parse_colon_pairs(value: str, *, env_var: str) -> dict[str, str]:
    """`"a:1,b:2"` -> `{"a": "1", "b": "2"}`. `env_var` is only used to
    name the offending variable in an error message."""
    pairs: dict[str, str] = {}
    for pair in value.split(","):
        pair = pair.strip()
        if not pair:
            continue
        left, _, right = pair.partition(":")
        if not left or not right:
            raise RuntimeError(f"{env_var} entry {pair!r} is not in 'left:right' form")
        pairs[left] = right
    return pairs


def parse_int_quotas(value: str, *, env_var: str) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for tenant_id, limit_str in parse_colon_pairs(value, env_var=env_var).items():
        try:
            quotas[tenant_id] = int(limit_str)
        except ValueError:
            raise RuntimeError(
                f"{env_var} entry {tenant_id!r}:{limit_str!r} has a non-integer limit"
            ) from None
    return quotas
