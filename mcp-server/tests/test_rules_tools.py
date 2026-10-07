"""mcp_server's Phase 10 rule tools (list_rules/match_rules/review_rule/
extract_rules), called directly via MCPServer.call_tool() against
in-memory test doubles — same conventions as test_server.py.
"""

from __future__ import annotations

import json

import pytest
from mcp.server.mcpserver.exceptions import UnexpectedToolError
from mcp_server.server import build_server

from ragcore.corpus import Corpus
from ragcore.rules.base import CandidateSpan, VerificationResult
from ragcore.rules.sqlite_store import SqliteRuleStore
from ragcore.rules.types import Effect, FactPredicate, Rule, rule_content_hash
from ragcore.types import Chunk, ChunkMeta


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
        return []

    def get(self, chunk_ids):
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    def get_by_source(self, source_id):
        return [c for c in self._chunks.values() if c.meta.source_id == source_id]

    def delete(self, *, source_id=None, chunk_ids=None):
        return 0

    def list_sources(self):
        from ragcore.types import SourceInfo

        by_source: dict[str, list] = {}
        for c in self._chunks.values():
            by_source.setdefault(c.meta.source_id, []).append(c)
        return [
            SourceInfo(
                source_id=sid, path=chunks[0].meta.source_path, title=chunks[0].meta.source_title,
                source_type=chunks[0].meta.source_type, chunk_count=len(chunks),
            )
            for sid, chunks in by_source.items()
        ]

    def count(self) -> int:
        return len(self._chunks)


class _FakeOntology:
    def normalize_fact_key(self, raw_mention: str) -> str | None:
        return "fact.always"

    def normalize_fact_value(self, fact_key: str, raw_mention: str):
        return 1


class _FakeDomainPack:
    domain = "test"
    pack_id = "p"
    pack_version = "1"
    ontology = _FakeOntology()

    def fact_types(self):
        return ()

    def extraction_system_prompt(self) -> str:
        return "extract"

    def verification_system_prompt(self) -> str:
        return "verify"


class _FakeDetector:
    model = "fake"

    def detect(self, chunks, *, domain_pack):
        chunk = chunks[0]
        return [CandidateSpan(chunk_id=chunk.chunk_id, source_id=chunk.meta.source_id, text=chunk.text)]


class _FakeExtractor:
    model = "fake"

    def extract(self, candidates, *, domain_pack):
        rules = []
        for c in candidates:
            cond = FactPredicate(fact="fact.always", operator="eq", value=1)
            effect = Effect(category="health", polarity="negative")
            rule_id = rule_content_hash(
                domain=domain_pack.domain, book_id=c.source_id, verbatim_span=c.text,
                conditions=cond, effect=effect,
            )
            rules.append(Rule(
                rule_id=rule_id, domain=domain_pack.domain, pack_id="p", pack_version="1",
                book_id=c.source_id, page_start=1, chunk_ids=(c.chunk_id,), verbatim_span=c.text,
                language="en", normalized_statement="n", conditions=cond, effect=effect,
                rule_type="HARD", confidence=0.9, extraction_model="fake", extraction_version="v1",
                created_at="now",
            ))
        return rules


class _FakeVerifier:
    model = "fake"

    def verify(self, rules, *, domain_pack):
        return [VerificationResult(rule_id=r.rule_id, supported=True, reason="ok") for r in rules]


def _result_dict(call_tool_result) -> dict:
    [content] = call_tool_result.content
    return json.loads(content.text)


def _rule(rule_id: str, *, review_status: str = "verified", book_id: str = "b1") -> Rule:
    return Rule(
        rule_id=rule_id, domain="test", pack_id="p", pack_version="1", book_id=book_id,
        page_start=1, chunk_ids=("c1",), verbatim_span="s", language="en", normalized_statement="n",
        conditions=FactPredicate(fact="x", operator="eq", value=8),
        effect=Effect(category="health", polarity="negative"), rule_type="HARD",
        confidence=0.9, extraction_model="m", extraction_version="v1", created_at="now",
        review_status=review_status,
    )


@pytest.fixture
def corpus() -> Corpus:
    return Corpus(_InMemoryStore(), _FakeEmbedder())


@pytest.fixture
def store() -> SqliteRuleStore:
    return SqliteRuleStore(":memory:")


async def test_list_rules(corpus, store) -> None:
    store.add_rules([_rule("r1")])
    server = build_server(corpus_provider=lambda: corpus, rule_store_provider=lambda: store)

    result = await server.call_tool("list_rules", {"domain": "test"})

    [rule] = [json.loads(c.text) for c in result.content]
    assert rule["rule_id"] == "r1"


async def test_match_rules(corpus, store) -> None:
    store.add_rules([_rule("r1")])
    server = build_server(corpus_provider=lambda: corpus, rule_store_provider=lambda: store)

    result = await server.call_tool("match_rules", {"domain": "test", "facts": {"x": 8}})

    body = _result_dict(result)
    assert [m["rule"]["rule_id"] for m in body["matched"]] == ["r1"]
    assert [r["rule_id"] for r in body["effective"]] == ["r1"]


async def test_match_rules_excludes_unverified_by_default(corpus, store) -> None:
    store.add_rules([_rule("r1", review_status="auto")])
    server = build_server(corpus_provider=lambda: corpus, rule_store_provider=lambda: store)

    result = await server.call_tool("match_rules", {"domain": "test", "facts": {"x": 8}})

    assert _result_dict(result)["matched"] == []


async def test_review_rule_approve(corpus, store) -> None:
    store.add_rules([_rule("r1", review_status="auto")])
    server = build_server(corpus_provider=lambda: corpus, rule_store_provider=lambda: store)

    result = await server.call_tool(
        "review_rule", {"rule_id": "r1", "action": "approve", "reviewer": "alice", "reason": "ok"}
    )

    assert _result_dict(result)["review_status"] == "verified"


async def test_review_rule_reject(corpus, store) -> None:
    store.add_rules([_rule("r1", review_status="auto")])
    server = build_server(corpus_provider=lambda: corpus, rule_store_provider=lambda: store)

    result = await server.call_tool(
        "review_rule", {"rule_id": "r1", "action": "reject", "reviewer": "alice", "reason": "unsupported"}
    )

    assert _result_dict(result)["review_status"] == "rejected"


async def test_review_rule_edit_creates_a_new_version(corpus, store) -> None:
    store.add_rules([_rule("r1")])
    server = build_server(corpus_provider=lambda: corpus, rule_store_provider=lambda: store)

    result = await server.call_tool(
        "review_rule",
        {"rule_id": "r1", "action": "edit", "reviewer": "carol", "reason": "bump", "confidence": 0.99},
    )

    body = _result_dict(result)
    assert body["rule_id"] != "r1"
    assert body["confidence"] == 0.99


async def test_review_rule_with_an_invalid_action_raises(corpus, store) -> None:
    store.add_rules([_rule("r1")])
    server = build_server(corpus_provider=lambda: corpus, rule_store_provider=lambda: store)

    with pytest.raises(UnexpectedToolError):
        await server.call_tool(
            "review_rule", {"rule_id": "r1", "action": "nonsense", "reviewer": "alice"}
        )


async def test_extract_rules(corpus, store) -> None:
    meta = ChunkMeta(
        source_id="src", source_path="book.md", source_title="Book", source_type="md",
        chapter_index=0, chapter_title="Ch 1", section=None, section_path=(), page_start=None,
        page_end=None, chunk_index=0, context_group_id="src:0", char_start=0, char_end=10, token_count=5,
    )
    corpus.store.upsert([Chunk(chunk_id="src:0", text="Some rule text here", vector=[0.1] * 8, meta=meta)])

    server = build_server(
        corpus_provider=lambda: corpus,
        rule_store_provider=lambda: store,
        domain_pack_registry_provider=lambda: {"test": _FakeDomainPack()},
        rule_extraction_pipeline_provider=lambda: (_FakeDetector(), _FakeExtractor(), _FakeVerifier()),
    )

    result = await server.call_tool("extract_rules", {"domain": "test"})

    body = _result_dict(result)
    assert body["rules_stored"] == 1
    assert len(store.rules_for_domain("test")) == 1


async def test_extract_rules_without_a_registered_domain_pack_raises(corpus, store) -> None:
    server = build_server(corpus_provider=lambda: corpus, rule_store_provider=lambda: store)

    with pytest.raises(UnexpectedToolError):
        await server.call_tool("extract_rules", {"domain": "unregistered"})
