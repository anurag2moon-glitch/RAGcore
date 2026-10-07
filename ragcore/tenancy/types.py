"""Data types for ragcore.tenancy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TenantDeletionReport:
    """Summary returned by TenantScopedCorpus.hard_delete()."""

    tenant_id: str
    source_ids: tuple[str, ...]
    chunks_deleted: int
