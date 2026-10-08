"""run_ingest_job, exercised end to end: a real moto-mocked S3 upload
(the same raw file the API would have uploaded before enqueuing, keyed
by the file's own content hash - exactly as `app.main`'s real /ingest
handler does, since `Corpus.ingest()` independently re-derives
`source_id` as `sha256(file bytes)` and the two must agree), downloaded
by the job, ingested into a real in-memory Corpus, with tenant_id
passed straight through to Corpus.ingest() - no TenantScopedCorpus
wrapping needed here, unlike app.rules_routes.extract_rules's own
systemic-gap workaround, since ingest() already has first-class
tenant_id support.

Also exercises the real RQ + fakeredis queue/worker lifecycle (enqueue
-> a real SimpleWorker processes it -> Job.fetch reads back status/
result) - run_ingest_job must stay importable by module path for this
to work at all (RQ serializes a job as a reference, not by value;
confirmed directly - a job defined inline in a throwaway script fails
immediately with "Functions from the __main__ module cannot be
processed by workers").
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import boto3
import fakeredis
import pytest
from app import worker
from app.jobs import fetch_job, get_redis_connection
from app.settings import Settings
from app.worker.ingest_job import run_ingest_job
from fakes import FakeEmbedder, InMemoryStore
from moto import mock_aws
from rq import Queue, SimpleWorker
from rq.job import Job

from ragcore.corpus import Corpus
from ragcore.raw_files.s3 import S3RawFileStore

BUCKET = "ragcore-test-bucket"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def raw_file_store():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield S3RawFileStore(BUCKET, prefix="raw-files/", client=client)


@pytest.fixture
def corpus() -> Corpus:
    return Corpus(InMemoryStore(), FakeEmbedder())


@pytest.fixture
def staged_book(tmp_path: Path, raw_file_store: S3RawFileStore) -> str:
    """Writes a book, uploads it to the mocked S3 store keyed by its
    own content hash (matching what the real API handler does before
    enqueuing), and returns that hash - the source_id Corpus.ingest()
    will independently re-derive once the job downloads and ingests
    it."""
    text = "# Chapter 1\n\nSome content about chunking and embeddings.\n"
    source_id = _content_hash(text)
    book = tmp_path / "book.md"
    book.write_text(text)
    raw_file_store.put(source_id, book)
    return source_id


@pytest.fixture
def patched_worker_providers(monkeypatch, raw_file_store, corpus, tmp_path):
    """run_ingest_job calls get_raw_file_store()/get_corpus()/
    tenant_data_path() directly (it runs inside an RQ worker process,
    not a FastAPI request, so there's no app.dependency_overrides to
    use) - patched at the point of use in app.worker.ingest_job,
    the standard seam for a plain function with no DI framework."""
    monkeypatch.setattr(worker.ingest_job, "get_raw_file_store", lambda: raw_file_store)
    monkeypatch.setattr(worker.ingest_job, "get_corpus", lambda: corpus)
    monkeypatch.setattr(
        worker.ingest_job, "tenant_data_path", lambda tenant_id: tmp_path / (tenant_id or "_shared")
    )


def test_run_ingest_job_downloads_and_ingests(
    patched_worker_providers, staged_book: str, corpus: Corpus
) -> None:
    result = run_ingest_job(source_id=staged_book, tenant_id="tenant-a")

    assert result["source_ids"] == [staged_book]
    assert result["chunks_ingested"] > 0
    assert any(s.source_id == staged_book for s in corpus.sources())


def test_run_ingest_job_tags_chunks_with_the_tenant_id(
    patched_worker_providers, staged_book: str, corpus: Corpus
) -> None:
    run_ingest_job(source_id=staged_book, tenant_id="tenant-a")

    chunks = corpus.store.get_by_source(staged_book)
    assert chunks
    assert all(c.meta.tenant_id == "tenant-a" for c in chunks)


def test_run_ingest_job_with_no_tenant_leaves_chunks_untagged(
    patched_worker_providers, staged_book: str, corpus: Corpus
) -> None:
    run_ingest_job(source_id=staged_book, tenant_id=None)

    chunks = corpus.store.get_by_source(staged_book)
    assert chunks
    assert all(c.meta.tenant_id is None for c in chunks)


def test_run_ingest_job_cleans_up_the_staged_local_file(
    patched_worker_providers, staged_book: str, tmp_path: Path
) -> None:
    run_ingest_job(source_id=staged_book, tenant_id="tenant-a")

    staged_dir = tmp_path / "tenant-a" / "incoming"
    assert list(staged_dir.iterdir()) == []


def test_run_ingest_job_cleans_up_even_on_ingest_failure(
    patched_worker_providers, staged_book: str, tmp_path: Path, monkeypatch, corpus: Corpus
) -> None:
    def _boom(path, **kwargs):
        raise RuntimeError("simulated ingest failure")

    monkeypatch.setattr(corpus, "ingest", _boom)

    with pytest.raises(RuntimeError, match="simulated ingest failure"):
        run_ingest_job(source_id=staged_book, tenant_id="tenant-a")

    staged_dir = tmp_path / "tenant-a" / "incoming"
    assert list(staged_dir.iterdir()) == []


# -- the real RQ + fakeredis queue/worker lifecycle -----------------------


@pytest.fixture
def fake_redis_connection():
    return fakeredis.FakeStrictRedis()


def test_enqueue_then_worker_processes_then_fetch_job_reads_the_result(
    patched_worker_providers, staged_book: str, fake_redis_connection,
) -> None:
    queue = Queue("ingest", connection=fake_redis_connection)
    job = queue.enqueue(run_ingest_job, source_id=staged_book, tenant_id="tenant-a")
    assert job.get_status() == "queued"

    rq_worker = SimpleWorker([queue], connection=fake_redis_connection)
    rq_worker.work(burst=True)

    finished = Job.fetch(job.id, connection=fake_redis_connection)
    assert finished.get_status() == "finished"
    assert finished.return_value()["source_ids"] == [staged_book]


def test_fetch_job_returns_none_for_an_unknown_job_id(monkeypatch, fake_redis_connection) -> None:
    monkeypatch.setattr("app.jobs.get_redis_connection", lambda: fake_redis_connection)
    assert fetch_job("nonexistent-job-id") is None


def test_fetch_job_returns_the_job_for_a_known_id(fake_redis_connection) -> None:
    queue = Queue("ingest", connection=fake_redis_connection)
    job = queue.enqueue(run_ingest_job, source_id="whatever", tenant_id=None)

    import app.jobs as jobs_module

    jobs_module.get_redis_connection.cache_clear()
    try:
        found = Job.fetch(job.id, connection=fake_redis_connection)
        assert found.id == job.id
    finally:
        jobs_module.get_redis_connection.cache_clear()


def test_get_redis_connection_requires_redis_url(monkeypatch) -> None:
    import app.jobs as jobs_module

    monkeypatch.setattr("app.jobs.get_settings", lambda: Settings(redis_url=None))
    jobs_module.get_redis_connection.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="REDIS_URL"):
            get_redis_connection()
    finally:
        jobs_module.get_redis_connection.cache_clear()
