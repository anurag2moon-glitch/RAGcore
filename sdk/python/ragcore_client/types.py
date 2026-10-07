"""Response types for ragcore_client.

Plain dataclasses, not a reuse of ragcore.types — this SDK talks to the
reference server's HTTP JSON contract (server/app/models.py), not to
ragcore's own in-process dataclasses, and a real third-party consumer of
this SDK never imports ragcore directly at all.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IngestResult:
    source_ids: list[str]
    chunks_ingested: int
    sources_replaced: list[str]
    errors: list[tuple[str, str]]


@dataclass(frozen=True)
class SourceInfo:
    source_id: str
    path: str
    title: str | None
    source_type: str
    chunk_count: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source_id: str
    source_path: str
    chapter_title: str | None
    tenant_id: str | None
    license_id: str | None


@dataclass(frozen=True)
class AskResult:
    text: str
    citations: list[RetrievedChunk]
