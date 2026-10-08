"""`GET /health/ready`'s actual dependency checks — separate from
`GET /health` (a pure liveness check: "is the process up," no I/O,
always `200`). Readiness checks only whichever dependency this
deployment actually configured; an unconfigured one is left out of
the report entirely, never reported as a failure.

Each check opens its own fresh, short-lived connection rather than
reusing a cached singleton (`app.jobs.get_redis_connection`, etc.) —
a readiness probe should answer "would a *new* connection attempt
succeed right now," not "does a cached connection object already
exist" (those can diverge: a connection cached before a dependency
went down would otherwise report healthy forever). Each check is
independent and best-effort: one dependency being unreachable doesn't
stop the others from being checked.
"""

from __future__ import annotations

from app.settings import get_settings


def _check_postgres(database_url: str) -> str:
    import psycopg

    try:
        with psycopg.connect(database_url, connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
        return "ok"
    except Exception as e:  # noqa: BLE001 - readiness reports any failure reason verbatim, not a specific type
        return f"error: {e}"


def _check_redis(redis_url: str) -> str:
    import redis

    try:
        redis.from_url(redis_url, socket_connect_timeout=3).ping()
        return "ok"
    except Exception as e:  # noqa: BLE001 - same as above
        return f"error: {e}"


def _check_corpus() -> str:
    from app.corpus_provider import get_corpus

    try:
        get_corpus()
        return "ok"
    except Exception as e:  # noqa: BLE001 - same as above
        return f"error: {e}"


def run_readiness_checks() -> dict[str, str]:
    settings = get_settings()
    checks: dict[str, str] = {}
    if settings.ragcore_db_path:
        checks["corpus"] = _check_corpus()
    if settings.database_url:
        checks["postgres"] = _check_postgres(settings.database_url)
    if settings.redis_url:
        checks["redis"] = _check_redis(settings.redis_url)
    return checks
