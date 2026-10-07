"""MCP server exposing a single ragcore.Corpus as tools: ingest_file,
retrieve, ask, list_sources.

`build_server(corpus_provider=...)` takes the Corpus provider as a
constructor-injectable callable — the same swappable-component
discipline every other part of this codebase family follows (compare
server/app/main.py's `Depends(get_corpus)`). Real usage calls
`build_server()` with no arguments (the default `get_corpus`, reading
`RAGCORE_DB_PATH`); tests pass their own provider returning an in-memory
test Corpus, so no real Voyage/LanceDB credentials are ever needed to
test this module.

Run: `python -m mcp_server.server` (stdio transport, the standard way
an MCP client like Claude Desktop/Code launches a local MCP server),
with RAGCORE_DB_PATH set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mcp.server.mcpserver import MCPServer

from mcp_server.corpus_provider import get_corpus
from ragcore.corpus import Corpus
from ragcore.types import Retrieved


def _chunk_to_dict(r: Retrieved) -> dict[str, Any]:
    return {
        "chunk_id": r.chunk.chunk_id,
        "text": r.chunk.text,
        "score": r.score,
        "source_id": r.chunk.meta.source_id,
        "source_path": r.chunk.meta.source_path,
        "chapter_title": r.chunk.meta.chapter_title,
        "tenant_id": r.chunk.meta.tenant_id,
        "license_id": r.chunk.meta.license_id,
    }


def build_server(*, corpus_provider: Callable[[], Corpus] = get_corpus) -> MCPServer:
    server = MCPServer(
        "ragcore",
        description="Single-tenant access to one ragcore.Corpus: ingest, retrieve, ask, list sources.",
    )

    @server.tool()
    def ingest_file(path: str) -> dict[str, Any]:
        """Ingest a file already on this machine (an MCP server runs
        locally, launched by its client, so a local path — unlike the
        HTTP server's /ingest — is the natural shape here, not a
        multipart upload)."""
        corpus = corpus_provider()
        report = corpus.ingest(Path(path))
        return {
            "source_ids": report.source_ids,
            "chunks_ingested": report.chunks_ingested,
            "sources_replaced": report.sources_replaced,
            "errors": report.errors,
        }

    @server.tool()
    def retrieve(query: str, k: int = 10) -> list[dict[str, Any]]:
        """Retrieve the k most relevant chunks for `query`."""
        corpus = corpus_provider()
        results = corpus.retrieve(query, k=k)
        return [_chunk_to_dict(r) for r in results]

    @server.tool()
    def ask(query: str, k: int = 10) -> dict[str, Any]:
        """Retrieve then synthesize an answer. Fails if this corpus has
        no Generator configured."""
        corpus = corpus_provider()
        if corpus.generator is None:
            raise ValueError("this corpus has no Generator configured; ask is unavailable")
        answer = corpus.ask(query, k=k)
        return {
            "text": answer.text,
            "citations": [_chunk_to_dict(r) for r in answer.citations],
        }

    @server.tool()
    def list_sources() -> list[dict[str, Any]]:
        """List every source currently ingested into this corpus."""
        corpus = corpus_provider()
        return [
            {
                "source_id": s.source_id,
                "path": s.path,
                "title": s.title,
                "source_type": s.source_type,
                "chunk_count": s.chunk_count,
            }
            for s in corpus.sources()
        ]

    return server


server = build_server()


if __name__ == "__main__":
    server.run()
