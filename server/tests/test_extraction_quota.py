"""Per-tenant extraction token quota (app.extraction_quota) - Phase 10
§8's "quotas apply to extraction jobs (token cost metering)". Unit
tests of the dependency functions directly (no HTTP layer needed to
exercise this logic in isolation), plus one end-to-end /rules/extract
test confirming the route actually rejects once a tenant is over
budget.
"""

from __future__ import annotations

import pytest
from app.auth import InMemoryApiKeyStore, get_api_key_store
from app.corpus_provider import get_corpus
from app.extraction_quota import (
    InMemoryExtractionTokenMeter,
    enforce_extraction_token_quota,
    get_extraction_token_meter,
    get_extraction_token_quotas,
    record_extraction_tokens,
)
from app.main import app
from app.rules_provider import (
    get_domain_pack_registry,
    get_rule_extraction_pipeline,
    get_rule_store,
)
from fakes import FakeEmbedder, InMemoryStore
from fastapi import HTTPException
from fastapi.testclient import TestClient

from ragcore.corpus import Corpus
from ragcore.embedding.cost import UsageTracker
from ragcore.rules.base import CandidateSpan, VerificationResult
from ragcore.rules.sqlite_store import SqliteRuleStore
from ragcore.types import Chunk, ChunkMeta

# -- InMemoryExtractionTokenMeter -----------------------------------------


def test_meter_starts_at_zero_for_an_unknown_tenant() -> None:
    meter = InMemoryExtractionTokenMeter()
    assert meter.tokens_used("tenant-a") == 0


def test_meter_accumulates_across_multiple_records() -> None:
    meter = InMemoryExtractionTokenMeter()
    meter.record("tenant-a", 100)
    meter.record("tenant-a", 50)
    assert meter.tokens_used("tenant-a") == 150


def test_meter_tracks_tenants_independently() -> None:
    meter = InMemoryExtractionTokenMeter()
    meter.record("tenant-a", 100)
    meter.record("tenant-b", 5)
    assert meter.tokens_used("tenant-a") == 100
    assert meter.tokens_used("tenant-b") == 5


# -- enforce_extraction_token_quota ---------------------------------------


def test_enforce_is_a_noop_in_single_tenant_mode() -> None:
    meter = InMemoryExtractionTokenMeter()
    meter.record("tenant-a", 999999)
    enforce_extraction_token_quota(meter=meter, tenant_id=None, quotas={"tenant-a": 1})


def test_enforce_is_a_noop_for_a_tenant_with_no_quota_entry() -> None:
    meter = InMemoryExtractionTokenMeter()
    meter.record("tenant-a", 999999)
    enforce_extraction_token_quota(meter=meter, tenant_id="tenant-a", quotas={})


def test_enforce_allows_a_tenant_under_quota() -> None:
    meter = InMemoryExtractionTokenMeter()
    meter.record("tenant-a", 500)
    enforce_extraction_token_quota(meter=meter, tenant_id="tenant-a", quotas={"tenant-a": 1000})


def test_enforce_rejects_a_tenant_at_or_over_quota() -> None:
    meter = InMemoryExtractionTokenMeter()
    meter.record("tenant-a", 1000)

    with pytest.raises(HTTPException) as exc_info:
        enforce_extraction_token_quota(meter=meter, tenant_id="tenant-a", quotas={"tenant-a": 1000})

    assert exc_info.value.status_code == 402


# -- record_extraction_tokens ----------------------------------------------


def test_record_extraction_tokens_is_a_noop_in_single_tenant_mode() -> None:
    meter = InMemoryExtractionTokenMeter()
    record_extraction_tokens(500, meter=meter, tenant_id=None)
    assert meter.tokens_used("tenant-a") == 0


def test_record_extraction_tokens_records_for_a_real_tenant() -> None:
    meter = InMemoryExtractionTokenMeter()
    record_extraction_tokens(500, meter=meter, tenant_id="tenant-a")
    assert meter.tokens_used("tenant-a") == 500


# -- end-to-end through /rules/extract -------------------------------------


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

    def __init__(self, *, tokens_per_call: int = 500) -> None:
        self.usage = UsageTracker()
        self._tokens_per_call = tokens_per_call

    def extract(self, candidates, *, domain_pack):
        self.usage.record_success(self._tokens_per_call)
        return []  # content doesn't matter for this test - only token cost does


class _FakeVerifier:
    model = "fake"

    def verify(self, rules, *, domain_pack):
        return [VerificationResult(rule_id=r.rule_id, supported=True, reason="ok") for r in rules]


@pytest.fixture
def extraction_quota_client():
    corpus = Corpus(InMemoryStore(), FakeEmbedder())
    meta = ChunkMeta(
        source_id="src", source_path="book.md", source_title="Book", source_type="md",
        chapter_index=0, chapter_title="Ch 1", section=None, section_path=(), page_start=None,
        page_end=None, chunk_index=0, context_group_id="src:0", char_start=0, char_end=10, token_count=5,
        tenant_id="tenant-a",
    )
    corpus.store.upsert([Chunk(chunk_id="src:0", text="Some rule text here", vector=[0.1] * 8, meta=meta)])
    store = SqliteRuleStore(":memory:")
    api_key_store = InMemoryApiKeyStore({"key-a": "tenant-a"})

    app.dependency_overrides[get_corpus] = lambda: corpus
    app.dependency_overrides[get_rule_store] = lambda: store
    app.dependency_overrides[get_domain_pack_registry] = lambda: {"test": _FakeDomainPack()}
    app.dependency_overrides[get_api_key_store] = lambda: api_key_store
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_extract_rules_records_real_token_usage_from_the_pipeline(extraction_quota_client) -> None:
    app.dependency_overrides[get_rule_extraction_pipeline] = lambda: (
        _FakeDetector(), _FakeExtractor(tokens_per_call=600), _FakeVerifier(),
    )

    resp = extraction_quota_client.post(
        "/rules/extract", params={"domain": "test"}, headers={"X-API-Key": "key-a"}
    )

    assert resp.status_code == 200
    assert resp.json()["total_tokens_used"] == 600


def test_extract_rules_rejects_before_running_once_a_tenant_is_already_over_quota(
    extraction_quota_client,
) -> None:
    """build_rules() checkpoints a chapter after one successful
    extraction, so repeated calls against unchanged content never
    re-run the pipeline or accrue further tokens - that's
    build_rules()'s own correct resumability, verified independently
    in tests/test_rules_pipeline.py, not something to re-prove here.
    This test instead directly proves the half this route is
    responsible for: a tenant already over quota from past jobs (an
    already-populated meter) is rejected *before* the pipeline ever
    runs, not after wastefully re-running it."""
    over_quota_meter = InMemoryExtractionTokenMeter()
    over_quota_meter.record("tenant-a", 1000)
    app.dependency_overrides[get_extraction_token_meter] = lambda: over_quota_meter
    app.dependency_overrides[get_rule_extraction_pipeline] = lambda: (
        _FakeDetector(), _FakeExtractor(tokens_per_call=600), _FakeVerifier(),
    )
    app.dependency_overrides[get_extraction_token_quotas] = lambda: {"tenant-a": 1000}

    resp = extraction_quota_client.post(
        "/rules/extract", params={"domain": "test"}, headers={"X-API-Key": "key-a"}
    )

    assert resp.status_code == 402
    # rejected before running - the meter's own total is unchanged
    assert over_quota_meter.tokens_used("tenant-a") == 1000
