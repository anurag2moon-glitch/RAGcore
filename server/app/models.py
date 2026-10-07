"""Pydantic request/response models for the reference API.

Deliberately a projection, not a 1:1 mirror of every ragcore.types
dataclass field — ChunkMeta alone has ~20 fields, and an HTTP API
contract should expose what a typical client actually needs, not
everything the library happens to carry internally. Extend these as a
real client's needs become concrete, rather than exhaustively mirroring
ChunkMeta speculatively now.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ragcore.generation.base import Answer
from ragcore.types import IngestReport, Retrieved, SourceInfo


class IngestResponse(BaseModel):
    source_ids: list[str]
    chunks_ingested: int
    sources_replaced: list[str]
    errors: list[tuple[str, str]]

    @classmethod
    def from_report(cls, report: IngestReport) -> IngestResponse:
        return cls(
            source_ids=report.source_ids,
            chunks_ingested=report.chunks_ingested,
            sources_replaced=report.sources_replaced,
            errors=report.errors,
        )


class SourceResponse(BaseModel):
    source_id: str
    path: str
    title: str | None
    source_type: str
    chunk_count: int

    @classmethod
    def from_source_info(cls, info: SourceInfo) -> SourceResponse:
        return cls(
            source_id=info.source_id,
            path=info.path,
            title=info.title,
            source_type=info.source_type,
            chunk_count=info.chunk_count,
        )


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    text: str
    score: float
    source_id: str
    source_path: str
    chapter_title: str | None
    tenant_id: str | None
    license_id: str | None

    @classmethod
    def from_retrieved(cls, r: Retrieved) -> RetrievedChunkResponse:
        return cls(
            chunk_id=r.chunk.chunk_id,
            text=r.chunk.text,
            score=r.score,
            source_id=r.chunk.meta.source_id,
            source_path=r.chunk.meta.source_path,
            chapter_title=r.chunk.meta.chapter_title,
            tenant_id=r.chunk.meta.tenant_id,
            license_id=r.chunk.meta.license_id,
        )


class RetrieveRequest(BaseModel):
    query: str
    k: int = 10
    filters: dict[str, Any] | None = None
    min_score: float | None = None


class RetrieveResponse(BaseModel):
    results: list[RetrievedChunkResponse]


class AskRequest(BaseModel):
    query: str
    k: int = 10
    filters: dict[str, Any] | None = None


class AskResponse(BaseModel):
    text: str
    citations: list[RetrievedChunkResponse]

    @classmethod
    def from_answer(cls, answer: Answer) -> AskResponse:
        return cls(
            text=answer.text,
            citations=[RetrievedChunkResponse.from_retrieved(r) for r in answer.citations],
        )


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    detail: str
