"""Phase 7 multi-tenancy: enforcement on top of the tenant_id foundation.

See docs/adr/0006-tenant-id-foundation-not-enforcement.md. `ChunkMeta.
tenant_id` (ragcore/types.py) and `Corpus.ingest(..., tenant_id=...)` /
`LanceDBStore`'s filterable fields (added in commit 7b7393d) make tenant-
scoped filtering *possible* but enforce nothing — a caller who forgets a
`filters={"tenant_id": ...}` on a `retrieve()`/`ask()` call silently reads
across every tenant in the store. `TenantScopedCorpus` closes that gap for
one `Corpus` instance bound to one fixed tenant: every method that reads
or writes always applies that tenant_id, and raises `TenancyError` rather
than silently ignoring a caller-supplied value that disagrees with it.
"""

from ragcore.tenancy.scoped_corpus import TenantScopedCorpus

__all__ = ["TenantScopedCorpus"]
