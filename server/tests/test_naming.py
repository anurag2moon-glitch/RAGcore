"""app.naming: tenant data paths, object storage keys, table names —
all derived from app.settings, none hardcoded at their call sites.
"""

from __future__ import annotations

from pathlib import Path

from app.naming import (
    API_KEYS_TABLE,
    TENANTS_TABLE,
    backup_object_key,
    raw_file_object_key,
    tenant_data_path,
)


def test_tenant_data_path_is_scoped_under_the_configured_root() -> None:
    path = tenant_data_path("tenant-a")
    assert path == Path("data/tenants") / "tenant-a"


def test_tenant_data_path_with_no_tenant_uses_a_shared_subdirectory() -> None:
    path = tenant_data_path(None)
    assert path == Path("data/tenants") / "_shared"


def test_tenant_data_paths_for_different_tenants_are_distinct() -> None:
    assert tenant_data_path("tenant-a") != tenant_data_path("tenant-b")


def test_raw_file_object_key_uses_the_configured_prefix() -> None:
    assert raw_file_object_key("src123") == "raw-files/src123"


def test_backup_object_key_uses_the_configured_prefix_and_tag() -> None:
    assert backup_object_key("2026-10-07", "postgres.dump") == "backups/2026-10-07/postgres.dump"


def test_table_names_are_stable_constants() -> None:
    assert TENANTS_TABLE == "tenants"
    assert API_KEYS_TABLE == "api_keys"
