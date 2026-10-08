"""The async ingestion job queue (ADR 0014) — a Redis connection and
one RQ `Queue`, both process-wide singletons built from
`app.settings.Settings`. `POST /ingest` (`app.main`) enqueues onto
this queue; the `worker` container's `rq worker ingest` consumes it,
running `app.worker.ingest_job.run_ingest_job`.

RQ, not a hand-rolled queue, specifically because its `Worker` already
implements the one behavior the deployment request asks for by name
("workers finish or checkpoint the current ingestion stage before
exiting") — confirmed directly from its source
(`rq.worker.Worker.request_stop`) during ADR 0014's own research: the
first SIGTERM triggers a "warm shutdown" that lets the in-flight job
finish, and only a second signal force-exits. Docker's own stop
sequence (SIGTERM, then SIGKILL after a grace period) maps onto this
exactly, with zero custom signal-handling code written here.
"""

from __future__ import annotations

from functools import lru_cache

import redis
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job

from app.settings import get_settings

INGEST_QUEUE_NAME = "ingest"


@lru_cache(maxsize=1)
def get_redis_connection() -> redis.Redis:
    settings = get_settings()
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL must be set to use async ingestion")
    return redis.from_url(settings.redis_url)


@lru_cache(maxsize=1)
def get_ingest_queue() -> Queue:
    return Queue(INGEST_QUEUE_NAME, connection=get_redis_connection())


def fetch_job(job_id: str) -> Job | None:
    try:
        return Job.fetch(job_id, connection=get_redis_connection())
    except NoSuchJobError:
        return None
