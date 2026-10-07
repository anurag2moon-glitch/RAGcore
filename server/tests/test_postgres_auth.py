"""PostgresApiKeyStore against a real Postgres server. Excluded by
default; run explicitly with `pytest -m postgres`. Requires
RAGCORE_TEST_POSTGRES_URL (a libpq connection string pointing at a
database with the tenants/api_keys schema already migrated - run
`alembic upgrade head` against it first) — skips cleanly when unset,
same convention ragcore's own tests/test_postgres_ingest_checkpoint_store.py
already uses for this exact kind of opt-in external dependency.

Each test uses its own uniquely-prefixed tenant_id/api_key so repeated
runs against a persistent database never collide with a previous run's
rows.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.postgres

CONNINFO = os.environ.get("RAGCORE_TEST_POSTGRES_URL")
pytest.importorskip("psycopg")


@pytest.fixture
def store():
    if not CONNINFO:
        pytest.skip("RAGCORE_TEST_POSTGRES_URL not set")
    from app.postgres_auth import PostgresApiKeyStore

    return PostgresApiKeyStore(CONNINFO)


@pytest.fixture
def tenant_id() -> str:
    return f"test-tenant-{uuid.uuid4()}"


@pytest.fixture
def api_key() -> str:
    return f"test-key-{uuid.uuid4()}"


def test_resolve_tenant_for_an_unknown_key_returns_none(store) -> None:
    assert store.resolve_tenant("definitely-not-a-real-key") is None


def test_create_tenant_and_api_key_then_resolve(store, tenant_id: str, api_key: str) -> None:
    store.create_tenant(tenant_id, display_name="Test Tenant")
    store.create_api_key(api_key, tenant_id=tenant_id)

    assert store.resolve_tenant(api_key) == tenant_id


def test_create_tenant_is_idempotent(store, tenant_id: str) -> None:
    store.create_tenant(tenant_id, display_name="First Name")
    store.create_tenant(tenant_id, display_name="Updated Name")  # must not raise


def test_revoked_api_key_no_longer_resolves(store, tenant_id: str, api_key: str) -> None:
    store.create_tenant(tenant_id, display_name="Test Tenant")
    store.create_api_key(api_key, tenant_id=tenant_id)
    assert store.resolve_tenant(api_key) == tenant_id

    store.revoke_api_key(api_key)

    assert store.resolve_tenant(api_key) is None


def test_recreating_a_revoked_key_un_revokes_it(store, tenant_id: str, api_key: str) -> None:
    store.create_tenant(tenant_id, display_name="Test Tenant")
    store.create_api_key(api_key, tenant_id=tenant_id)
    store.revoke_api_key(api_key)
    assert store.resolve_tenant(api_key) is None

    store.create_api_key(api_key, tenant_id=tenant_id)

    assert store.resolve_tenant(api_key) == tenant_id


def test_a_key_can_be_reassigned_to_a_different_tenant(store, api_key: str) -> None:
    tenant_a = f"test-tenant-a-{uuid.uuid4()}"
    tenant_b = f"test-tenant-b-{uuid.uuid4()}"
    store.create_tenant(tenant_a, display_name="A")
    store.create_tenant(tenant_b, display_name="B")
    store.create_api_key(api_key, tenant_id=tenant_a)
    assert store.resolve_tenant(api_key) == tenant_a

    store.create_api_key(api_key, tenant_id=tenant_b)

    assert store.resolve_tenant(api_key) == tenant_b
