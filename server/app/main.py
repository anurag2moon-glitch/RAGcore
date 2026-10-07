"""Reference FastAPI service exposing a single ragcore.Corpus over HTTP.
See docs/adr/0008-api-server-as-sibling-package.md for the sibling-package
design, and app/auth.py's own docstring for the opt-in multi-tenant
auth layer (RAGCORE_API_KEYS unset = single-tenant, exactly as before
that module existed).

Run: uvicorn app.main:app --reload (from server/, with RAGCORE_DB_PATH set).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.corpus_provider import get_scoped_corpus
from app.models import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    RetrievedChunkResponse,
    RetrieveRequest,
    RetrieveResponse,
    SourceResponse,
)
from ragcore.corpus import Corpus
from ragcore.errors import RagcoreError, TenancyError
from ragcore.tenancy import TenantScopedCorpus

logger = logging.getLogger("ragcore_server")

ScopedCorpus = Corpus | TenantScopedCorpus

app = FastAPI(
    title="ragcore reference server",
    description=(
        "HTTP surface over one ragcore.Corpus. Single-tenant and unauthenticated "
        "unless RAGCORE_API_KEYS is set, in which case every route is scoped via "
        "ragcore.tenancy.TenantScopedCorpus. See docs/adr/0008."
    ),
    version="0.1.0",
)


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
    return HealthResponse(status="ok")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile,
    corpus: ScopedCorpus = Depends(get_scoped_corpus),  # noqa: B008 - FastAPI's own DI idiom
) -> IngestResponse:
    """Multipart file upload — not a server-local path, since accepting
    only a path a remote caller can't actually control would defeat the
    point of an HTTP ingest endpoint. Saved to a temp file preserving
    the original filename's extension (ragcore's LoaderRegistry
    dispatches by extension, see ragcore/loaders/__init__.py)."""
    suffix = Path(file.filename or "").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        report = corpus.ingest(Path(tmp.name))
    return IngestResponse.from_report(report)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    req: RetrieveRequest,
    corpus: ScopedCorpus = Depends(get_scoped_corpus),  # noqa: B008 - FastAPI's own DI idiom
) -> RetrieveResponse:
    results = corpus.retrieve(req.query, k=req.k, filters=req.filters, min_score=req.min_score)
    return RetrieveResponse(results=[RetrievedChunkResponse.from_retrieved(r) for r in results])


@app.post("/ask", response_model=AskResponse)
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


@app.get("/sources", response_model=list[SourceResponse])
def list_sources(
    corpus: ScopedCorpus = Depends(get_scoped_corpus),  # noqa: B008 - FastAPI's own DI idiom
) -> list[SourceResponse]:
    return [SourceResponse.from_source_info(s) for s in corpus.sources()]


@app.delete("/sources/{source_id}")
def delete_source(
    source_id: str,
    corpus: ScopedCorpus = Depends(get_scoped_corpus),  # noqa: B008 - FastAPI's own DI idiom
) -> dict[str, int]:
    deleted = corpus.remove(source_id)
    return {"chunks_deleted": deleted}
