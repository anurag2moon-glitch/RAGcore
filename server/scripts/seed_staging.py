"""Seeds one staging tenant and one API key into Postgres — the
minimal-staging-deployment request's own explicit ask: "Seed script
for a staging tenant and API key." Idempotent (safe to run more than
once): `PostgresApiKeyStore.create_tenant`/`create_api_key` are both
upserts.

Requires the `tenants`/`api_keys` tables to already exist (run
`alembic upgrade head` first — the one-shot `migrate` Compose service
does this automatically before `api`/`worker` start).

Usage (from server/, with DATABASE_URL set):
    python scripts/seed_staging.py [tenant_id] [api_key]

Both arguments are optional — omitted, they default to "staging" and
a freshly-generated key (printed to stdout, since a generated secret
has nowhere else to go on first run).
"""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.postgres_auth import PostgresApiKeyStore
from app.settings import get_settings


def main() -> None:
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL must be set to run this script")

    tenant_id = sys.argv[1] if len(sys.argv) > 1 else "staging"
    api_key = sys.argv[2] if len(sys.argv) > 2 else f"staging-{secrets.token_urlsafe(24)}"

    store = PostgresApiKeyStore(settings.database_url)
    store.create_tenant(tenant_id, display_name=f"{tenant_id} (seeded)")
    store.create_api_key(api_key, tenant_id=tenant_id)

    print(f"Seeded tenant {tenant_id!r} with API key: {api_key}")
    print("Use it as: Authorization: Bearer <key>  (or)  X-API-Key: <key>")


if __name__ == "__main__":
    main()
