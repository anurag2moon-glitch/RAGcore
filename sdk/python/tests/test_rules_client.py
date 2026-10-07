"""RagcoreClient's Phase 10 rule methods, tested the same way
test_client.py tests everything else: FastAPI's TestClient wrapping
the real server/ app object in-process, Corpus/RuleStore dependencies
overridden with in-memory doubles, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_SERVER_DIR = Path(__file__).resolve().parents[3] / "server"
sys.path.insert(0, str(_SERVER_DIR))

from app.corpus_provider import get_corpus
from app.main import app
from app.rules_provider import get_rule_store

from ragcore.corpus import Corpus
from ragcore.rules.sqlite_store import SqliteRuleStore
from ragcore.rules.types import Effect, FactPredicate, Rule

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragcore_client import RagcoreClient


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
        return []

    def get_by_source(self, source_id):
        return []

    def delete(self, *, source_id=None, chunk_ids=None):
        return 0

    def list_sources(self):
        return []

    def count(self) -> int:
        return len(self._chunks)


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
def rules_client():
    corpus = Corpus(_InMemoryStore(), _FakeEmbedder())
    store = SqliteRuleStore(":memory:")
    app.dependency_overrides[get_corpus] = lambda: corpus
    app.dependency_overrides[get_rule_store] = lambda: store
    client = RagcoreClient(http_client=TestClient(app))
    yield client, store
    client.close()
    app.dependency_overrides.clear()


def test_list_rules(rules_client) -> None:
    client, store = rules_client
    store.add_rules([_rule("r1")])

    rules = client.list_rules("test")

    assert [r.rule_id for r in rules] == ["r1"]
    assert rules[0].conditions == {"kind": "predicate", "fact": "x", "operator": "eq", "value": 8}


def test_list_rules_scoped_to_a_book(rules_client) -> None:
    client, store = rules_client
    store.add_rules([_rule("r1", book_id="book1"), _rule("r2", book_id="book2")])

    rules = client.list_rules("test", book_id="book1")

    assert [r.rule_id for r in rules] == ["r1"]


def test_match_rules(rules_client) -> None:
    client, store = rules_client
    store.add_rules([_rule("r1")])

    result = client.match_rules("test", {"x": 8})

    assert [m.rule.rule_id for m in result.matched] == ["r1"]
    assert [r.rule_id for r in result.effective] == ["r1"]


def test_match_rules_include_unverified(rules_client) -> None:
    client, store = rules_client
    store.add_rules([_rule("r1", review_status="auto")])

    assert client.match_rules("test", {"x": 8}).matched == []
    result = client.match_rules("test", {"x": 8}, include_unverified=True)
    assert [m.rule.rule_id for m in result.matched] == ["r1"]


def test_approve_and_reject_rule(rules_client) -> None:
    client, store = rules_client
    store.add_rules([_rule("r1", review_status="auto"), _rule("r2", review_status="auto", book_id="b2")])

    approved = client.approve_rule("r1", reviewer="alice", reason="looks right")
    rejected = client.reject_rule("r2", reviewer="alice", reason="unsupported")

    assert approved.review_status == "verified"
    assert rejected.review_status == "rejected"


def test_edit_rule_creates_a_new_version(rules_client) -> None:
    client, store = rules_client
    store.add_rules([_rule("r1")])

    edited = client.edit_rule("r1", reviewer="carol", reason="bump confidence", confidence=0.99)

    assert edited.rule_id != "r1"
    assert edited.confidence == 0.99
    assert edited.version == 2


def test_errors_surface_as_ragcore_client_error(rules_client) -> None:
    from ragcore_client import RagcoreClientError

    client, _store = rules_client
    with pytest.raises(RagcoreClientError) as exc_info:
        client.approve_rule("nonexistent", reviewer="alice")
    assert exc_info.value.status_code == 404
