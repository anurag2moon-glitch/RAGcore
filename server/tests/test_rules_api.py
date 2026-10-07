"""Phase 10 §8 rule-serving REST routes: list_rules, match_rules,
approve/reject/edit, extract_rules, and tenant isolation for private
packs. FastAPI's TestClient, no network, no real Claude/Voyage
credentials — the extraction pipeline and domain pack are injected via
dependency overrides, the same pattern test_api.py already uses for
Corpus/Generator.
"""

from __future__ import annotations

import pytest
from app.auth import InMemoryApiKeyStore, get_api_key_store
from app.corpus_provider import get_corpus
from app.main import app
from app.rules_provider import (
    get_domain_pack_registry,
    get_rule_extraction_pipeline,
    get_rule_store,
)
from fakes import FakeEmbedder, InMemoryStore
from fastapi.testclient import TestClient

from ragcore.corpus import Corpus
from ragcore.rules.base import CandidateSpan, VerificationResult
from ragcore.rules.sqlite_store import SqliteRuleStore
from ragcore.rules.types import Effect, FactPredicate, Rule, rule_content_hash
from ragcore.types import Chunk, ChunkMeta


def _rule(
    rule_id: str, *, domain: str = "test", book_id: str = "b1",
    tenant_scope: str = "global", tenant_id: str | None = None, review_status: str = "verified",
) -> Rule:
    return Rule(
        rule_id=rule_id, domain=domain, pack_id="p", pack_version="1", book_id=book_id,
        page_start=1, chunk_ids=("c1",), verbatim_span="s", language="en", normalized_statement="n",
        conditions=FactPredicate(fact="x", operator="eq", value=8),
        effect=Effect(category="health", polarity="negative"), rule_type="HARD",
        confidence=0.9, extraction_model="m", extraction_version="v1", created_at="now",
        review_status=review_status, tenant_scope=tenant_scope, tenant_id=tenant_id,
    )


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


@pytest.fixture
def store() -> SqliteRuleStore:
    return SqliteRuleStore(":memory:")


@pytest.fixture
def client(store: SqliteRuleStore) -> TestClient:
    corpus = Corpus(InMemoryStore(), FakeEmbedder())
    app.dependency_overrides[get_corpus] = lambda: corpus
    app.dependency_overrides[get_rule_store] = lambda: store
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# -- list_rules -------------------------------------------------------


def test_list_rules_returns_rules_for_the_domain(client: TestClient, store: SqliteRuleStore) -> None:
    store.add_rules([_rule("r1")])

    resp = client.get("/rules", params={"domain": "test"})

    assert resp.status_code == 200
    assert [r["rule_id"] for r in resp.json()] == ["r1"]


def test_list_rules_scoped_to_a_book(client: TestClient, store: SqliteRuleStore) -> None:
    store.add_rules([_rule("r1", book_id="book1"), _rule("r2", book_id="book2")])

    resp = client.get("/rules", params={"domain": "test", "book_id": "book1"})

    assert [r["rule_id"] for r in resp.json()] == ["r1"]


def test_list_rules_excludes_another_domains_private_rule_without_auth(
    client: TestClient, store: SqliteRuleStore
) -> None:
    store.add_rules([
        _rule("r_global", tenant_scope="global"),
        _rule("r_private", tenant_scope="tenant", tenant_id="tenant-a", book_id="b2"),
    ])

    resp = client.get("/rules", params={"domain": "test"})

    assert [r["rule_id"] for r in resp.json()] == ["r_global"]


# -- match_rules --------------------------------------------------------


def test_match_rules_returns_matched_and_effective(client: TestClient, store: SqliteRuleStore) -> None:
    store.add_rules([_rule("r1")])

    resp = client.post("/rules/match", json={"domain": "test", "facts": {"x": 8}})

    assert resp.status_code == 200
    body = resp.json()
    assert [m["rule"]["rule_id"] for m in body["matched"]] == ["r1"]
    assert [r["rule_id"] for r in body["effective"]] == ["r1"]


def test_match_rules_excludes_unverified_by_default(client: TestClient, store: SqliteRuleStore) -> None:
    store.add_rules([_rule("r1", review_status="auto")])

    resp = client.post("/rules/match", json={"domain": "test", "facts": {"x": 8}})

    assert resp.json()["matched"] == []


def test_match_rules_include_unverified_opts_in(client: TestClient, store: SqliteRuleStore) -> None:
    store.add_rules([_rule("r1", review_status="auto")])

    resp = client.post(
        "/rules/match", json={"domain": "test", "facts": {"x": 8}, "include_unverified": True}
    )

    assert [m["rule"]["rule_id"] for m in resp.json()["matched"]] == ["r1"]


# -- review: approve / reject / edit -------------------------------


def test_approve_rule(client: TestClient, store: SqliteRuleStore) -> None:
    store.add_rules([_rule("r1", review_status="auto")])

    resp = client.post("/rules/r1/approve", json={"reviewer": "alice", "reason": "looks right"})

    assert resp.status_code == 200
    assert resp.json()["review_status"] == "verified"


def test_reject_rule_requires_a_reason(client: TestClient, store: SqliteRuleStore) -> None:
    store.add_rules([_rule("r1", review_status="auto")])

    resp = client.post("/rules/r1/reject", json={"reviewer": "alice", "reason": ""})

    assert resp.status_code == 400


def test_reject_rule(client: TestClient, store: SqliteRuleStore) -> None:
    store.add_rules([_rule("r1", review_status="auto")])

    resp = client.post("/rules/r1/reject", json={"reviewer": "alice", "reason": "unsupported"})

    assert resp.status_code == 200
    assert resp.json()["review_status"] == "rejected"


def test_edit_rule_creates_a_new_version(client: TestClient, store: SqliteRuleStore) -> None:
    store.add_rules([_rule("r1")])

    resp = client.post(
        "/rules/r1/edit", json={"reviewer": "carol", "reason": "bump", "confidence": 0.99}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["rule_id"] != "r1"
    assert body["confidence"] == 0.99
    assert body["version"] == 2


def test_edit_rule_requires_a_reason(client: TestClient, store: SqliteRuleStore) -> None:
    store.add_rules([_rule("r1")])

    resp = client.post("/rules/r1/edit", json={"reviewer": "carol", "reason": "", "confidence": 0.99})

    assert resp.status_code == 400


def test_review_action_on_an_unknown_rule_returns_404(client: TestClient) -> None:
    resp = client.post("/rules/nonexistent/approve", json={"reviewer": "alice", "reason": "x"})
    assert resp.status_code == 404


def test_review_action_on_another_tenants_private_rule_returns_404_without_auth(
    client: TestClient, store: SqliteRuleStore
) -> None:
    store.add_rules([_rule("r_private", tenant_scope="tenant", tenant_id="tenant-a")])

    resp = client.post("/rules/r_private/approve", json={"reviewer": "alice", "reason": "x"})

    assert resp.status_code == 404


# -- extract_rules --------------------------------------------------------


@pytest.fixture
def extraction_client(store: SqliteRuleStore) -> TestClient:
    corpus = Corpus(InMemoryStore(), FakeEmbedder())
    meta = ChunkMeta(
        source_id="src", source_path="book.md", source_title="Book", source_type="md",
        chapter_index=0, chapter_title="Ch 1", section=None, section_path=(), page_start=None,
        page_end=None, chunk_index=0, context_group_id="src:0", char_start=0, char_end=10, token_count=5,
    )
    corpus.store.upsert([Chunk(chunk_id="src:0", text="Some rule text here", vector=[0.1] * 8, meta=meta)])

    app.dependency_overrides[get_corpus] = lambda: corpus
    app.dependency_overrides[get_rule_store] = lambda: store
    app.dependency_overrides[get_domain_pack_registry] = lambda: {"test": _FakeDomainPack()}
    app.dependency_overrides[get_rule_extraction_pipeline] = lambda: (
        _FakeDetector(), _FakeExtractor(), _FakeVerifier(),
    )
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_extract_rules_runs_the_pipeline_and_stores_results(
    extraction_client: TestClient, store: SqliteRuleStore
) -> None:
    resp = extraction_client.post("/rules/extract", params={"domain": "test"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["rules_stored"] == 1
    assert len(store.rules_for_domain("test")) == 1


def test_extract_rules_without_a_registered_domain_pack_returns_404(client: TestClient) -> None:
    resp = client.post("/rules/extract", params={"domain": "unregistered_domain"})
    assert resp.status_code == 404


def test_extract_rules_without_a_configured_pipeline_returns_503(
    client: TestClient, store: SqliteRuleStore
) -> None:
    app.dependency_overrides[get_domain_pack_registry] = lambda: {"test": _FakeDomainPack()}
    resp = client.post("/rules/extract", params={"domain": "test"})
    assert resp.status_code == 503


# -- tenant isolation across the whole router --------------------------


@pytest.fixture
def tenant_client(store: SqliteRuleStore) -> TestClient:
    corpus = Corpus(InMemoryStore(), FakeEmbedder())
    api_key_store = InMemoryApiKeyStore({"key-a": "tenant-a", "key-b": "tenant-b"})
    app.dependency_overrides[get_corpus] = lambda: corpus
    app.dependency_overrides[get_rule_store] = lambda: store
    app.dependency_overrides[get_api_key_store] = lambda: api_key_store
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_tenant_a_cannot_see_tenant_bs_private_rules(
    tenant_client: TestClient, store: SqliteRuleStore
) -> None:
    store.add_rules([
        _rule("r_global", tenant_scope="global"),
        _rule("r_tenant_a", tenant_scope="tenant", tenant_id="tenant-a", book_id="b2"),
        _rule("r_tenant_b", tenant_scope="tenant", tenant_id="tenant-b", book_id="b3"),
    ])

    resp_a = tenant_client.get("/rules", params={"domain": "test"}, headers={"X-API-Key": "key-a"})
    resp_b = tenant_client.get("/rules", params={"domain": "test"}, headers={"X-API-Key": "key-b"})

    assert {r["rule_id"] for r in resp_a.json()} == {"r_global", "r_tenant_a"}
    assert {r["rule_id"] for r in resp_b.json()} == {"r_global", "r_tenant_b"}


def test_a_request_with_no_api_key_is_rejected_once_auth_is_enabled(tenant_client: TestClient) -> None:
    resp = tenant_client.get("/rules", params={"domain": "test"})
    assert resp.status_code == 401
