# 0020: Backup and restore for Postgres, the data volume, and object storage

## Status

Accepted.

## Context

The deployment task asked for "backup script and restore script for
Postgres, LanceDB data and object storage, with a tested restore."

## Decisions

**All three backups land in the same S3/MinIO bucket, under
`app.naming.backup_object_key` (`{s3_backups_prefix}{tag}/{name}`) -
not a separate host-mounted directory.** `app.naming.backup_object_key`
already existed before this unit, which settled this design early:
centralizing in object storage means the backup/restore scripts need
only network access to Postgres and S3 (both already reachable
wherever the script runs) plus read access to the mounted data
volume - no extra bind mount, no separate credentials, one storage
system to manage for a "minimal staging" deployment. The scripts are
meant to run inside the `api`/`worker` image (`docker compose exec api
python scripts/backup.py <tag>`), which already has both.

**Postgres is dumped table-by-table via `psycopg`'s `COPY ... (FORMAT
BINARY)`, not the `pg_dump` binary.** `python:3.12-slim` doesn't ship
`pg_dump`, and installing `postgresql-client` just for this would be
an extra package in an image that otherwise needs none; `psycopg` is
already a dependency, and this server's Postgres schema (ADR 0016) is
two small registry tables (`tenants`, `api_keys`), not a schema `
pg_dump`'s generality is needed for. Restore `TRUNCATE`s each table
before loading the backup back in - a restore replaces current state
with the backup's state, it doesn't merge the two.

**The data-dir backup uses fixed, semantic archive names
(`lancedb`/`rules_sqlite`/`tenant_data` - `scripts/backup.py
data_dir_roles()`), not each path's own basename.** A path's basename
is deployment-specific; restoring into an environment whose current
settings name these paths differently must still work, since the
backup is an opaque disaster-recovery artifact, not something
replayed only on the exact machine that produced it.

**Verified with a real round-trip, not just reviewed: the data-dir and
object-storage halves.** `tests/test_backup_restore.py` runs
`_backup_data_dir`/`_restore_data_dir` and
`_backup_raw_files`/`_restore_raw_files` for real against a
moto-mocked S3 bucket and real temp directories - simulates total data
loss (`shutil.rmtree`/`unlink`/`delete_object`) between backup and
restore, then asserts the exact original bytes come back. This is the
"tested restore" the task asked for, as literally as this environment
allows.

**The Postgres half is `pytest.mark.postgres`-gated, not executed by
default - the same precedent ADR 0005/0016 already established** for
every Postgres-touching test in this server, since no live Postgres
server is available in this development environment. It is reviewed
and exercised for import/syntax correctness by every other test in the
file running at all, and is runnable for real with
`RAGCORE_TEST_POSTGRES_URL` set.

**A moto-mocked S3 client intercepts only AWS's own default endpoints,
not a custom `endpoint_url`** (confirmed directly: a first test draft
using `Settings(s3_endpoint_url=...)` end to end made real network
calls that failed to connect, rather than being intercepted) - so
`get_raw_file_store_provider.get_s3_client` itself is monkeypatched in
tests, the same pattern `tests/conftest.py`'s existing
`_async_ingest_infra` fixture already uses for exactly this reason.

**`app.raw_files_provider` gained `get_s3_client()`, split out from
`get_raw_file_store()`.** `S3RawFileStore`'s own interface is keyed by
`source_id` with no listing/arbitrary-key operations -
`scripts/backup.py`/`restore.py` need `list_objects_v2`,
`upload_fileobj`/`download_fileobj`, and arbitrary keys under
`s3_backups_prefix`, none of which that interface covers. Exposing the
one underlying `boto3` client directly avoids a second,
backup-specific S3 abstraction.

## Consequences

- No automated backup schedule exists yet - `make backup TAG=...` is
  an operator-invoked action, matching "minimal staging, single node"
  (a cron-triggered schedule would be a reasonable next step, not
  needed for this pass).
- Restoring Postgres is all-or-nothing per table (`TRUNCATE` then
  reload) - there is no selective "restore just this one tenant" path.
- Backups live in the same bucket they're backing up pieces of (the
  raw-files prefix) - a real disaster that destroys the whole MinIO
  volume takes the backups with it. For actual staging use, the bucket
  itself should be mirrored off-node (e.g. `mc mirror` to a second
  location) - orthogonal to, and not covered by, this ADR's mechanism.
