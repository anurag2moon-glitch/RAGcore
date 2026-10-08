"""Shared fixtures for exercising the async /ingest -> GET /jobs/{id}
flow (ADR 0014) from a test, without each of test_api.py/test_auth.py/
test_usage.py re-implementing the same moto-S3 + fakeredis + worker-
wiring setup independently.

`run_ingest_job` (the RQ job function) calls `get_raw_file_store()`/
`get_corpus()` directly, not through FastAPI's DI container — so
overriding `app.dependency_overrides[get_corpus]` (which only affects
how *this request* resolves `Depends(get_corpus)`) has no effect on
what the job sees when a worker actually runs it. `ingest_sync` below
takes the `corpus` the job should use as an explicit argument (not a
same-named fixture it reaches for implicitly) specifically so a test
using a *different* Corpus per client variant (e.g.
`client_with_generator`) can't accidentally wire the job to the
*wrong* instance — that exact bug was caught here, by running the
first version of this fixture before trusting it, not assumed safe.
"""

from __future__ import annotations

import boto3
import fakeredis
import pytest
from app import worker
from app.jobs import get_ingest_queue, get_redis_connection
from app.raw_files_provider import get_raw_file_store
from fastapi.testclient import TestClient
from moto import mock_aws
from rq import SimpleWorker

from ragcore.corpus import Corpus
from ragcore.raw_files.s3 import S3RawFileStore

BUCKET = "ragcore-test-bucket"


@pytest.fixture
def _async_ingest_infra(monkeypatch):
    """The moto S3 bucket + fakeredis connection, with no Corpus
    wiring yet — `ingest_sync` binds the worker's view of
    `get_corpus()` per call, since which Corpus is "the" one under
    test varies per test."""
    with mock_aws():
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=BUCKET)
        raw_store = S3RawFileStore(BUCKET, prefix="raw-files/", client=s3_client)

        redis_conn = fakeredis.FakeStrictRedis()

        get_raw_file_store.cache_clear()
        get_ingest_queue.cache_clear()
        get_redis_connection.cache_clear()
        monkeypatch.setattr("app.main.get_raw_file_store", lambda: raw_store)
        monkeypatch.setattr("app.jobs.get_redis_connection", lambda: redis_conn)
        monkeypatch.setattr(worker.ingest_job, "get_raw_file_store", lambda: raw_store)

        yield raw_store, redis_conn

    get_raw_file_store.cache_clear()
    get_ingest_queue.cache_clear()
    get_redis_connection.cache_clear()


@pytest.fixture
def ingest_sync(_async_ingest_infra, monkeypatch):
    """`ingest_sync(client, filename, content, corpus, *, headers=None)
    -> requests.Response` — POSTs the upload, points the worker's
    `get_corpus()` at exactly the `corpus` instance passed in (so a
    test's own assertions and the job's own writes are guaranteed to
    hit the same object, never a second, independently-overridden
    one), runs a real RQ `SimpleWorker` burst inline, and returns the
    **final** `GET /jobs/{job_id}` response — not the `202` enqueue
    response — so a caller keeps asserting on
    `resp.json()["result"][...]` rather than juggling two responses."""
    _raw_store, redis_conn = _async_ingest_infra

    def _ingest(
        client: TestClient, filename: str, content: bytes, corpus: Corpus,
        *, headers: dict | None = None,
    ):
        monkeypatch.setattr(worker.ingest_job, "get_corpus", lambda: corpus)

        enqueue_resp = client.post(
            "/ingest", files={"file": (filename, content)}, headers=headers or {},
        )
        if enqueue_resp.status_code != 202:
            return enqueue_resp

        job_id = enqueue_resp.json()["job_id"]
        queue = get_ingest_queue()
        rq_worker = SimpleWorker([queue], connection=redis_conn)
        rq_worker.work(burst=True)

        return client.get(f"/jobs/{job_id}", headers=headers or {})

    return _ingest
