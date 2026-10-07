"""mcp_server.server: tool registration and behavior, called directly
via MCPServer.call_tool() (in-process, no stdio transport, no real MCP
client) against an in-memory test Corpus injected via build_server's
corpus_provider parameter.

MCPServer.call_tool() returns the tool's result JSON-serialized as a
single TextContent block (confirmed directly, not assumed — there is no
convert_result=False escape hatch at this level) and, separately,
re-raises a tool's own exception as mcp.server.mcpserver.exceptions.
UnexpectedToolError wrapping the original (also confirmed directly) -
not packaged as an is_error=True result the way going over an actual
stdio/JSON-RPC transport would. Tests account for both.
"""

from __future__ import annotations

import json

import pytest
from mcp.server.mcpserver.exceptions import UnexpectedToolError
from mcp_server.server import build_server

from ragcore.corpus import Corpus
from ragcore.generation.base import Answer
from ragcore.types import Retrieved, SourceInfo


class _FakeEmbedder:
    dim = 8
    model = "fake-embedder"

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * self.dim

    def embed_groups(self, groups, *, input_type="document"):
        return [[[0.1] * self.dim for _ in g] for g in groups]


class _InMemoryStore:
    dim = 8

    def __init__(self) -> None:
        self._chunks: dict[str, object] = {}

    def upsert(self, chunks) -> None:
        for c in chunks:
            self._chunks[c.chunk_id] = c

    def query(self, vector, k, filters=None):
        return [Retrieved(chunk=c, score=0.9) for c in list(self._chunks.values())[:k]]

    def get(self, chunk_ids):
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    def get_by_source(self, source_id):
        return [c for c in self._chunks.values() if c.meta.source_id == source_id]

    def delete(self, *, source_id=None, chunk_ids=None):
        if source_id is not None:
            to_remove = [cid for cid, c in self._chunks.items() if c.meta.source_id == source_id]
        else:
            to_remove = list(chunk_ids)
        for cid in to_remove:
            del self._chunks[cid]
        return len(to_remove)

    def list_sources(self):
        by_source: dict[str, list] = {}
        for c in self._chunks.values():
            by_source.setdefault(c.meta.source_id, []).append(c)
        return [
            SourceInfo(
                source_id=sid,
                path=chunks[0].meta.source_path,
                title=chunks[0].meta.source_title,
                source_type=chunks[0].meta.source_type,
                chunk_count=len(chunks),
            )
            for sid, chunks in by_source.items()
        ]

    def count(self) -> int:
        return len(self._chunks)


class _FakeGenerator:
    model = "fake-generator"

    def generate(self, query, context) -> Answer:
        return Answer(text=f"answer to {query!r}", citations=list(context))


def _result_values(call_tool_result) -> list:
    """A tool returning a dict yields exactly one TextContent block; a
    tool returning a list yields one TextContent block *per list item*
    (confirmed directly, not assumed — MCPServer serializes a list
    result this way, not as one combined JSON array)."""
    return [json.loads(c.text) for c in call_tool_result.content]


def _result_dict(call_tool_result) -> dict:
    [value] = _result_values(call_tool_result)
    return value


@pytest.fixture
def corpus() -> Corpus:
    return Corpus(_InMemoryStore(), _FakeEmbedder())


@pytest.fixture
def corpus_with_generator() -> Corpus:
    return Corpus(_InMemoryStore(), _FakeEmbedder(), generator=_FakeGenerator())


async def test_ingest_file_then_list_sources_then_retrieve(corpus, tmp_path) -> None:
    server = build_server(corpus_provider=lambda: corpus)
    book = tmp_path / "book.md"
    book.write_text("# Chapter 1\n\nSome content about chunking and embeddings.\n")

    ingest_result = await server.call_tool("ingest_file", {"path": str(book)})
    report = _result_dict(ingest_result)
    assert report["chunks_ingested"] > 0

    sources_result = await server.call_tool("list_sources", {})
    sources = _result_values(sources_result)
    assert any(s["source_id"] == report["source_ids"][0] for s in sources)

    retrieve_result = await server.call_tool("retrieve", {"query": "chunking", "k": 5})
    chunks = _result_values(retrieve_result)
    assert chunks
    assert all("chunk_id" in c for c in chunks)


async def test_ask_without_a_generator_raises(corpus) -> None:
    server = build_server(corpus_provider=lambda: corpus)
    with pytest.raises(UnexpectedToolError):
        await server.call_tool("ask", {"query": "anything"})


async def test_ask_with_a_generator_returns_an_answer(corpus_with_generator, tmp_path) -> None:
    server = build_server(corpus_provider=lambda: corpus_with_generator)
    book = tmp_path / "book.md"
    book.write_text("# Chapter 1\n\nSome content about chunking.\n")
    await server.call_tool("ingest_file", {"path": str(book)})

    ask_result = await server.call_tool("ask", {"query": "chunking"})

    body = _result_dict(ask_result)
    assert "answer to" in body["text"]
    assert body["citations"]


async def test_the_original_four_corpus_tools_are_registered(corpus) -> None:
    server = build_server(corpus_provider=lambda: corpus)
    tools = await server.list_tools()
    names = {t.name for t in tools}
    # Phase 10 added four more (list_rules/match_rules/review_rule/
    # extract_rules, see test_rules_tools.py) - this only pins the
    # original corpus-level four are still exactly these four names.
    assert {"ingest_file", "retrieve", "ask", "list_sources"} <= names
    assert names == {
        "ingest_file", "retrieve", "ask", "list_sources",
        "list_rules", "match_rules", "review_rule", "extract_rules",
    }
