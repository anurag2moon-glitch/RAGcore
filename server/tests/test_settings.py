"""app.settings.Settings: env var name mapping (unchanged from before
this model existed), defaults, and fail-fast validation on every
malformed-value path. Constructs Settings directly with explicit
kwargs throughout, never via the cached get_settings() — that cache is
process-wide and shared with every other test module, so a test
depending on a *specific* value would be order-dependent; an explicit
Settings(...) call always validates fresh, which is what every one of
these tests actually wants to exercise.
"""

from __future__ import annotations

import pytest
from app.settings import Settings
from pydantic import ValidationError


def test_env_var_names_match_the_pre_existing_names() -> None:
    """These exact names must keep working - they are this server's
    public configuration surface, documented in docs/CONFIGURATION.md
    and server/README.md before this settings model existed."""
    s = Settings(
        RAGCORE_DB_PATH="/tmp/x.lancedb",
        RAGCORE_RULES_DB_PATH="/tmp/x.sqlite3",
        RAGCORE_API_KEYS="k:t",
        RAGCORE_RATE_LIMIT_PER_MINUTE="60",
        RAGCORE_TENANT_QUOTAS="t:100",
        RAGCORE_EXTRACTION_TOKEN_QUOTAS="t:100",
    )
    assert s.ragcore_db_path == "/tmp/x.lancedb"
    assert s.ragcore_rules_db_path == "/tmp/x.sqlite3"
    assert s.ragcore_api_keys == "k:t"
    assert s.ragcore_rate_limit_per_minute == 60
    assert s.ragcore_tenant_quotas == "t:100"
    assert s.ragcore_extraction_token_quotas == "t:100"


def test_defaults_when_nothing_is_set() -> None:
    s = Settings(_env_file=None)
    assert s.environment == "dev"
    assert s.log_level == "INFO"
    assert s.ragcore_db_path is None
    assert s.s3_bucket == "ragcore"
    assert s.s3_raw_files_prefix == "raw-files/"
    assert s.s3_backups_prefix == "backups/"
    assert s.database_url is None
    assert s.redis_url is None


def test_rejects_a_non_positive_rate_limit() -> None:
    with pytest.raises(ValidationError, match="positive integer"):
        Settings(ragcore_rate_limit_per_minute=0)
    with pytest.raises(ValidationError, match="positive integer"):
        Settings(ragcore_rate_limit_per_minute=-1)


def test_rejects_malformed_api_keys() -> None:
    with pytest.raises(ValidationError, match="not in 'left:right' form"):
        Settings(ragcore_api_keys="no-colon")


def test_rejects_malformed_tenant_quotas() -> None:
    with pytest.raises(ValidationError, match="non-integer limit"):
        Settings(ragcore_tenant_quotas="tenant-a:not-a-number")


def test_rejects_malformed_extraction_token_quotas() -> None:
    with pytest.raises(ValidationError, match="non-integer limit"):
        Settings(ragcore_extraction_token_quotas="tenant-a:not-a-number")


def test_rejects_a_database_url_with_the_wrong_scheme() -> None:
    with pytest.raises(ValidationError, match="postgresql://"):
        Settings(database_url="mysql://user:pass@host/db")


def test_accepts_both_postgres_url_schemes() -> None:
    Settings(database_url="postgresql://user:pass@host/db")
    Settings(database_url="postgres://user:pass@host/db")


def test_rejects_a_redis_url_with_the_wrong_scheme() -> None:
    with pytest.raises(ValidationError, match="redis://"):
        Settings(redis_url="http://localhost:6379")


def test_accepts_both_redis_url_schemes() -> None:
    Settings(redis_url="redis://localhost:6379/0")
    Settings(redis_url="rediss://localhost:6379/0")


def test_rejects_an_unknown_environment_value() -> None:
    with pytest.raises(ValidationError):
        Settings(environment="production")  # must be exactly "prod", not "production"


def test_secret_read_from_a_mounted_secrets_file(tmp_path) -> None:
    """"Secrets read from env or a mounted secrets file" - a real file
    under secrets_dir named after the field populates it when no env
    var of the same name is set. A local subclass with its own
    model_config is used rather than monkeypatching Settings' own
    model_config (a class-level dict shared with every other test in
    this module) or RAGCORE_SECRETS_DIR (read once, at import time -
    too late to affect an already-imported Settings class)."""
    from pydantic_settings import SettingsConfigDict

    (tmp_path / "database_url").write_text("postgresql://secretuser:secretpw@db/ragcore")

    class SettingsWithSecretsDir(Settings):
        model_config = SettingsConfigDict(secrets_dir=str(tmp_path), extra="ignore")

    s = SettingsWithSecretsDir(_env_file=None)
    assert s.database_url == "postgresql://secretuser:secretpw@db/ragcore"


def test_env_var_still_wins_over_a_secrets_file(tmp_path) -> None:
    from pydantic_settings import SettingsConfigDict

    (tmp_path / "database_url").write_text("postgresql://from-file/db")

    class SettingsWithSecretsDir(Settings):
        model_config = SettingsConfigDict(secrets_dir=str(tmp_path), extra="ignore")

    s = SettingsWithSecretsDir(_env_file=None, database_url="postgresql://from-env/db")
    assert s.database_url == "postgresql://from-env/db"
