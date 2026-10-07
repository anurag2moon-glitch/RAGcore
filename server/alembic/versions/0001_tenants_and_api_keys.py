"""tenants and api_keys tables

Revision ID: 0001
Revises:
Create Date: 2026-10-07

Raw SQL throughout, no ORM models — matches every other store in this
codebase family (ragcore's own SQLite/Postgres stores are all raw SQL,
never an ORM); Alembic is used here purely for migration
*versioning/sequencing*, not schema-from-models generation. See
docs/adr/0016-postgres-tenant-registry-and-migrations.md.

Table names are written literally here (`tenants`, `api_keys`), not
imported from `app.naming.TENANTS_TABLE`/`API_KEYS_TABLE` — a migration
is a frozen historical record of what was actually run; importing a
live application constant into it would mean this file's behavior
could silently change if that constant were ever edited later, which
is the opposite of what a migration is for. `app.naming`'s constants
exist so `PostgresApiKeyStore` and the seed script can't drift from
*each other*; this migration is the one place those names originate,
by design, not a third thing that must be kept in sync with them.
"""

from __future__ import annotations

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE tenants (
            tenant_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE api_keys (
            api_key TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            revoked_at TIMESTAMPTZ
        )
        """
    )
    op.execute("CREATE INDEX idx_api_keys_tenant_id ON api_keys(tenant_id)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS api_keys")
    op.execute("DROP TABLE IF EXISTS tenants")
