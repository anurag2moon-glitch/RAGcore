"""`run_ingest_job` — the one function the `worker` container's `rq
worker ingest` process actually executes per job (ADR 0014). Must
stay importable by module path (`app.worker.ingest_job.run_ingest_job`)
— RQ serializes a job as a reference to this path, not by value, so
it can't be a closure or anything defined inline.

Downloads the raw file from object storage by `source_id` (the API's
`/ingest` handler already uploaded it there before enqueuing — see
`app.main`), stages it under a tenant-scoped local path
(`app.naming.tenant_data_path`, not a bare system temp dir — see that
function's own docstring for why), calls the **unmodified**
`Corpus.ingest()` with `tenant_id` passed straight through (ragcore's
own write-time tenant tagging, `ragcore.types.ChunkMeta.tenant_id` —
no `TenantScopedCorpus` wrapping needed here at all, unlike
`app.rules_routes.extract_rules`'s own systemic-gap workaround,
because `ingest()` already has first-class `tenant_id=...` support).
Always removes the staged local file afterward, success or failure.
"""

from __future__ import annotations

from typing import Any

from app.corpus_provider import get_corpus
from app.naming import tenant_data_path
from app.raw_files_provider import get_raw_file_store


def run_ingest_job(*, source_id: str, tenant_id: str | None) -> dict[str, Any]:
    store = get_raw_file_store()
    original_filename = store.original_filename(source_id) or source_id
    local_dir = tenant_data_path(tenant_id) / "incoming"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / f"{source_id}-{original_filename}"

    try:
        store.get(source_id, local_path)
        corpus = get_corpus()
        report = corpus.ingest(local_path, tenant_id=tenant_id)
        return {
            "source_ids": report.source_ids,
            "chunks_ingested": report.chunks_ingested,
            "sources_replaced": report.sources_replaced,
            "errors": report.errors,
        }
    finally:
        local_path.unlink(missing_ok=True)
