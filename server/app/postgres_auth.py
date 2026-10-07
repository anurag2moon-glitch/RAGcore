"""PostgresApiKeyStore: the real, persistent ApiKeyStore implementation
for anything beyond local dev — `app.auth.InMemoryApiKeyStore`'s own
docstring named exactly this as the intended replacement ("a real
deployment managing its own key lifecycle... against its own backing
store"). Implements `app.auth.ApiKeyStore` unchanged; nothing else in
`app.auth` or any route needs to change for this swap.

Schema (`app.naming.TENANTS_TABLE`/`API_KEYS_TABLE`) is managed by
Alembic (`server/alembic/`), not created here — unlike
`ragcore.ingestion.PostgresIngestCheckpointStore`, which creates its
own schema on connect (a library component with no migration tooling
of its own). This module assumes the tables already exist; `migrate`
(the one-shot Compose service) is what's responsible for that having
already happened before `api`/`worker` start.

**Tested only against a real Postgres, opt-in** (`pytest -m postgres`,
`RAGCORE_TEST_POSTGRES_URL`) — same reasoning and same honesty
disclaimer as `ragcore.ingestion.postgres_store`'s own module
docstring: written and reviewed against psycopg 3's documented API and
standard PostgreSQL syntax, but treat as unverified by actual
execution until run against a live Postgres server at least once.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.naming import API_KEYS_TABLE, TENANTS_TABLE

try:
    import psycopg
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    psycopg = None
else:
    _IMPORT_ERROR = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PostgresApiKeyStore:
    """`conninfo` is a standard libpq connection string
    (`postgresql://user:pass@host/db` — `app.settings.Settings.database_url`);
    `client` lets a caller (or a test, once a real Postgres is
    available) inject an already-open `psycopg.Connection` instead."""

    def __init__(self, conninfo: str = "", *, client: Any | None = None) -> None:
        if client is None and psycopg is None:
            raise RuntimeError(
                "PostgresApiKeyStore requires the 'psycopg[binary]' package "
                "(server/pyproject.toml's own dependency, not ragcore's)"
            ) from _IMPORT_ERROR
        self._conn = client or psycopg.connect(conninfo, autocommit=True)

    def resolve_tenant(self, api_key: str) -> str | None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT tenant_id FROM {API_KEYS_TABLE} "
                f"WHERE api_key = %s AND revoked_at IS NULL",
                (api_key,),
            )
            row = cur.fetchone()
        return row[0] if row else None

    # -- admin operations (used by the seed script and, later, any --
    # -- future tenant-management endpoint) --------------------------

    def create_tenant(self, tenant_id: str, *, display_name: str) -> None:
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {TENANTS_TABLE} (tenant_id, display_name, created_at) "
                f"VALUES (%s, %s, %s) "
                f"ON CONFLICT (tenant_id) DO UPDATE SET display_name = excluded.display_name",
                (tenant_id, display_name, _now()),
            )

    def create_api_key(self, api_key: str, *, tenant_id: str) -> None:
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {API_KEYS_TABLE} (api_key, tenant_id, created_at, revoked_at) "
                f"VALUES (%s, %s, %s, NULL) "
                f"ON CONFLICT (api_key) DO UPDATE SET "
                f"tenant_id = excluded.tenant_id, revoked_at = NULL",
                (api_key, tenant_id, _now()),
            )

    def revoke_api_key(self, api_key: str) -> None:
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                f"UPDATE {API_KEYS_TABLE} SET revoked_at = %s WHERE api_key = %s",
                (_now(), api_key),
            )
