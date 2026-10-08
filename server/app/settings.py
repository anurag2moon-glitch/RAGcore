"""The single validated settings model every provider module in this
server reads from — Phase "minimal staging deployment"'s own rule:
"All settings via environment variables with a validated settings
model. Fail fast on missing or invalid config." Replaces the scattered
`os.environ.get(...)` + `@lru_cache` pattern every provider module
used before this existed (`app.corpus_provider`, `app.auth`,
`app.rate_limit`, `app.quota`, `app.rules_provider`,
`app.extraction_quota`) — same environment variable *names* as before,
so nothing about how this server is configured changes externally,
only where that configuration is read from and validated.

**Fail fast means constructed once, eagerly, at import time** — see
`get_settings()`'s own docstring. A field-level validator error
(`pydantic.ValidationError`) or one of this module's own
`@field_validator`s re-raising a malformed-value error happens the
moment `app.main` is imported (i.e. the moment `uvicorn app.main:app`
starts), not on the first request that happens to touch the
misconfigured field — the gap the old per-provider `@lru_cache`
pattern had (`RAGCORE_DB_PATH` only ever failed when `/ingest` was
first called).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.env_parsing import parse_colon_pairs, parse_int_quotas

# "Secrets read from env or a mounted secrets file, never baked into
# images or committed" (the deployment request's own rule) -
# pydantic-settings' own `secrets_dir` does exactly this: for any field,
# a file named after it (lowercase - e.g. `database_url`) under this
# directory is read as that field's value, with a real env var of the
# same name still taking priority when both are present (confirmed
# directly: constructing Settings with both DATABASE_URL set and a
# matching secrets file present returns the env var's value). Standard
# Docker/Compose-secrets convention is `/run/secrets`; only passed to
# SettingsConfigDict when that directory actually exists, since
# pydantic-settings otherwise emits a UserWarning on every local/test
# run where it's absent - checked once at import time via a plain
# os.environ read (RAGCORE_SECRETS_DIR), not a pydantic field, since
# model_config itself is fixed at class-definition time.
_secrets_dir = os.environ.get("RAGCORE_SECRETS_DIR", "/run/secrets")
_SECRETS_DIR = _secrets_dir if Path(_secrets_dir).is_dir() else None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", secrets_dir=_SECRETS_DIR
    )

    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # -- ragcore storage (existing env var names, unchanged) --------------
    ragcore_db_path: str | None = None
    ragcore_rules_db_path: str | None = None

    # -- auth / rate limiting / quotas (existing env var names, unchanged) --
    ragcore_api_keys: str | None = None
    ragcore_rate_limit_per_minute: int | None = None
    ragcore_tenant_quotas: str | None = None
    ragcore_extraction_token_quotas: str | None = None

    # -- postgres: the new tenant/API-key registry (ADR 0014) -------------
    database_url: str | None = None

    # -- redis: the async ingest job queue (ADR 0014) ----------------------
    redis_url: str | None = None

    # -- object storage: MinIO/S3, raw files + backups (ADR 0014) ---------
    s3_endpoint_url: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_region: str = "us-east-1"
    s3_bucket: str = "ragcore"
    s3_raw_files_prefix: str = "raw-files/"
    s3_backups_prefix: str = "backups/"

    # -- tenant data paths (not hardcoded anywhere else) -------------------
    tenant_data_root: str = "./data/tenants"

    # -- observability ------------------------------------------------------
    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "ragcore-server"

    @field_validator("ragcore_rate_limit_per_minute")
    @classmethod
    def _rate_limit_positive(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("RAGCORE_RATE_LIMIT_PER_MINUTE must be a positive integer")
        return v

    @field_validator("ragcore_api_keys")
    @classmethod
    def _api_keys_well_formed(cls, v: str | None) -> str | None:
        if v:
            try:
                parse_colon_pairs(v, env_var="RAGCORE_API_KEYS")
            except RuntimeError as e:
                raise ValueError(str(e)) from e
        return v

    @field_validator("ragcore_tenant_quotas")
    @classmethod
    def _tenant_quotas_well_formed(cls, v: str | None) -> str | None:
        if v:
            try:
                parse_int_quotas(v, env_var="RAGCORE_TENANT_QUOTAS")
            except RuntimeError as e:
                raise ValueError(str(e)) from e
        return v

    @field_validator("ragcore_extraction_token_quotas")
    @classmethod
    def _extraction_quotas_well_formed(cls, v: str | None) -> str | None:
        if v:
            try:
                parse_int_quotas(v, env_var="RAGCORE_EXTRACTION_TOKEN_QUOTAS")
            except RuntimeError as e:
                raise ValueError(str(e)) from e
        return v

    @field_validator("database_url")
    @classmethod
    def _database_url_scheme(cls, v: str | None) -> str | None:
        if v and not v.startswith(("postgresql://", "postgres://")):
            raise ValueError("DATABASE_URL must be a postgresql:// (or postgres://) connection string")
        return v

    @field_validator("redis_url")
    @classmethod
    def _redis_url_scheme(cls, v: str | None) -> str | None:
        if v and not v.startswith(("redis://", "rediss://")):
            raise ValueError("REDIS_URL must be a redis:// (or rediss://) connection string")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached, but every call constructs-and-validates identically —
    the cache exists so every caller shares one instance, not to skip
    validation. `app.main` calls this once at module import time
    specifically so a bad `.env`/environment fails the process before
    it ever binds a port, rather than on the first request that
    touches the broken field."""
    return Settings()
