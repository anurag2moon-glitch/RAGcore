"""Reference FastAPI service exposing a single ragcore.Corpus over HTTP.
See docs/adr/0008-api-server-as-sibling-package.md for the sibling-package
design, and app/auth.py's own docstring for the opt-in multi-tenant
auth layer (RAGCORE_API_KEYS unset = single-tenant, exactly as before
that module existed).

Run: uvicorn app.main:app --reload (from server/, with RAGCORE_DB_PATH set).
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from app.auth import get_tenant_id
from app.corpus_provider import get_scoped_corpus
from app.jobs import fetch_job, get_ingest_queue
from app.logging_config import configure_logging, request_id_middleware
from app.metrics import get_request_metrics, metrics_middleware
from app.models import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
    IngestJobEnqueuedResponse,
    IngestResponse,
    JobStatusResponse,
    ReadinessResponse,
    RetrievedChunkResponse,
    RetrieveRequest,
    RetrieveResponse,
    SourceResponse,
)
from app.quota import enforce_quota
from app.rate_limit import enforce_rate_limit
from app.raw_files_provider import get_raw_file_store
from app.readiness import run_readiness_checks
from app.rules_routes import router as rules_router
from app.settings import get_settings
from app.usage import get_my_usage, record_usage
from app.worker.ingest_job import run_ingest_job
from ragcore.corpus import Corpus
from ragcore.errors import RagcoreError, TenancyError
from ragcore.tenancy import TenantScopedCorpus

logger = logging.getLogger("ragcore_server")

ScopedCorpus = Corpus | TenantScopedCorpus

# Fail fast: validates every setting (malformed RAGCORE_API_KEYS/
# RAGCORE_TENANT_QUOTAS/.../a non-postgresql:// DATABASE_URL/a non-
# redis:// REDIS_URL/...) the moment this module is imported - i.e.
# the moment `uvicorn app.main:app` starts - rather than on whichever
# future request first happens to touch the broken field. Every field
# has a safe default, so this never fails in a test process (which
# sets nothing) or in local dev with no .env at all.
_settings = get_settings()
configure_logging(_settings.log_level)

app = FastAPI(
    title="ragcore reference server",
    description=(
        "HTTP surface over one ragcore.Corpus. Single-tenant and unauthenticated "
        "unless RAGCORE_API_KEYS is set, in which case every route is scoped via "
        "ragcore.tenancy.TenantScopedCorpus. See docs/adr/0008."
    ),
    version="0.1.0",
)

app.middleware("http")(metrics_middleware)
app.middleware("http")(request_id_middleware)

app.include_router(rules_router)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Plain Prometheus text exposition - see app/metrics.py's own
    docstring for why this is hand-written rather than a dependency,
    and how it differs from the OTLP metrics FastAPI's own telemetry
    already exports to the collector."""
    return get_request_metrics().render_prometheus_text()


@app.exception_handler(TenancyError)
async def tenancy_error_handler(_request, exc: TenancyError) -> JSONResponse:
    logger.warning("TenancyError: %s", exc)
    return JSONResponse(status_code=403, content=ErrorResponse(detail=str(exc)).model_dump())


@app.exception_handler(RagcoreError)
async def ragcore_error_handler(_request, exc: RagcoreError) -> JSONResponse:
    logger.warning("RagcoreError: %s", exc)
    return JSONResponse(status_code=400, content=ErrorResponse(detail=str(exc)).model_dump())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness only: "is the process up and able to handle a
    request" — no I/O, never fails once the process has started.
    Kept as the pre-existing path (nothing that already depends on
    `/health` meaning liveness needs to change); `/health/live` is the
    same check under the more explicit, Kubernetes-convention name."""
    return HealthResponse(status="ok")


@app.get("/health/live", response_model=HealthResponse)
def health_live() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/health/ready", response_model=ReadinessResponse)
def health_ready() -> ReadinessResponse:
    """Readiness: actually reaches whichever of Postgres/Redis/the
    corpus's own vector store this deployment configured. `503` (via
    the raised HTTPException below) if any *configured* dependency's
    check failed — an unconfigured one is absent from `checks`
    entirely and never counts as a failure, so a minimal deployment
    with none of them set is trivially "ready"."""
    checks = run_readiness_checks()
    if any(v != "ok" for v in checks.values()):
        raise HTTPException(
            status_code=503, detail=ReadinessResponse(status="not ready", checks=checks).model_dump()
        )
    return ReadinessResponse(status="ready", checks=checks)


@app.post(
    "/ingest", response_model=IngestJobEnqueuedResponse, status_code=202,
    dependencies=[Depends(enforce_rate_limit), Depends(enforce_quota), Depends(record_usage)],
)
async def ingest(
    file: UploadFile,
    tenant_id: str | None = Depends(get_tenant_id),
) -> IngestJobEnqueuedResponse:
    """As of ADR 0014, ingestion is an async job, not run inline in
    this request: the uploaded file is content-hashed (the same
    `sha256(file bytes)` scheme every ragcore `Loader` independently
    re-derives as `source_id` — the two must agree, since the worker
    never sees the original filename-based identity, only this hash),
    uploaded to object storage under that hash, and a job referencing
    it is enqueued. `GET /jobs/{job_id}` polls for the result. Requires
    `REDIS_URL`/`S3_*` to be configured (ADR 0014) — fails with a clear
    `503` otherwise, rather than silently falling back to the old
    synchronous path (see that ADR's own "Consequences" for why there
    is deliberately no such fallback)."""
    content = await file.read()
    source_id = hashlib.sha256(content).hexdigest()

    try:
        store = get_raw_file_store()
        queue = get_ingest_queue()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    suffix = Path(file.filename or "").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(content)
        tmp.flush()
        store.put(source_id, Path(tmp.name))

    job = queue.enqueue(run_ingest_job, source_id=source_id, tenant_id=tenant_id)
    return IngestJobEnqueuedResponse(job_id=job.id, source_id=source_id)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    try:
        job = fetch_job(job_id)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job found with job_id {job_id!r}")

    status = job.get_status()
    result = IngestResponse(**job.return_value()) if status == "finished" else None
    error = None
    if status == "failed":
        latest = job.latest_result()
        error = latest.exc_string if latest is not None else "job failed (no result recorded)"
    return JobStatusResponse(job_id=job.id, status=status, result=result, error=error)


@app.post("/retrieve", response_model=RetrieveResponse, dependencies=[Depends(enforce_rate_limit), Depends(enforce_quota), Depends(record_usage)])
def retrieve(
    req: RetrieveRequest,
    corpus: ScopedCorpus = Depends(get_scoped_corpus),  # noqa: B008 - FastAPI's own DI idiom
) -> RetrieveResponse:
    results = corpus.retrieve(req.query, k=req.k, filters=req.filters, min_score=req.min_score)
    return RetrieveResponse(results=[RetrievedChunkResponse.from_retrieved(r) for r in results])


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(enforce_rate_limit), Depends(enforce_quota), Depends(record_usage)])
def ask(
    req: AskRequest,
    corpus: ScopedCorpus = Depends(get_scoped_corpus),  # noqa: B008 - FastAPI's own DI idiom
) -> AskResponse:
    if corpus.generator is None:
        raise HTTPException(
            status_code=400,
            detail="this corpus has no Generator configured; /ask is unavailable",
        )
    answer = corpus.ask(req.query, k=req.k, filters=req.filters)
    return AskResponse.from_answer(answer)


@app.get("/sources", response_model=list[SourceResponse], dependencies=[Depends(enforce_rate_limit), Depends(enforce_quota), Depends(record_usage)])
def list_sources(
    corpus: ScopedCorpus = Depends(get_scoped_corpus),  # noqa: B008 - FastAPI's own DI idiom
) -> list[SourceResponse]:
    return [SourceResponse.from_source_info(s) for s in corpus.sources()]


@app.delete("/sources/{source_id}", dependencies=[Depends(enforce_rate_limit), Depends(enforce_quota), Depends(record_usage)])
def delete_source(
    source_id: str,
    corpus: ScopedCorpus = Depends(get_scoped_corpus),  # noqa: B008 - FastAPI's own DI idiom
) -> dict[str, int]:
    deleted = corpus.remove(source_id)
    return {"chunks_deleted": deleted}


@app.get("/usage")
def usage(counts: dict[str, int] = Depends(get_my_usage)) -> dict[str, int]:  # noqa: B008
    """Request counts per endpoint for the calling tenant only — `400`
    in single-tenant mode (no tenant_id to meter by). Deliberately not
    rate-limited/metered itself — reading your own usage shouldn't
    count against it."""
    return counts
