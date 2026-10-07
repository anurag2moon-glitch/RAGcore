"""Alembic environment script. Reads DATABASE_URL from
app.settings.get_settings() (not from alembic.ini's own
sqlalchemy.url, which is deliberately left blank) — the connection
string goes through the exact same validated Settings model every
other provider in this server uses, never a second, independently-
configured copy.

Uses a SQLAlchemy Engine purely as Alembic's own transport (the
`postgresql+psycopg` dialect, SQLAlchemy 2.x's native psycopg **3**
support — not psycopg2, which this project doesn't install). No ORM
models anywhere: every migration under alembic/versions/ is raw SQL
via `op.execute(...)`, matching every other store in this codebase
family.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from app.settings import get_settings
from sqlalchemy import create_engine

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def _database_url() -> str:
    url = get_settings().database_url
    if not url:
        raise RuntimeError(
            "DATABASE_URL must be set to run migrations (e.g. "
            "postgresql://ragcore:ragcore@localhost:5432/ragcore)"
        )
    # Settings validates this already starts with postgresql:// or
    # postgres:// - rewritten to SQLAlchemy's psycopg-3-specific dialect
    # name so create_engine() picks psycopg (v3), not psycopg2 (not
    # installed in this project at all).
    return url.replace("postgresql://", "postgresql+psycopg://").replace(
        "postgres://", "postgresql+psycopg://"
    )


def run_migrations_online() -> None:
    engine = create_engine(_database_url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
