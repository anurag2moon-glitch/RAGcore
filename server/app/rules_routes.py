"""Phase 10 §8's "API endpoints": list_rules, match_rules, review_rule
(approve/reject/edit), extract_rules — the same four operations
mcp-server/'s MCP tools expose, as plain REST routes. A separate
APIRouter (not flat in app.main) since this is a genuinely separate
capability with its own provider module, mirroring how app.main stays
focused on the retrieval/generation surface it already had.

Tenant isolation for private packs (Phase 10 §8's own open item):
every route here filters by `tenant_id` resolved from
`app.auth.get_tenant_id` exactly like `app.corpus_provider.get_scoped_corpus`
does — a caller sees every `tenant_scope="global"` rule plus only
their *own* `tenant_scope="tenant"` rules, never another tenant's
private ones. This is enforced here, in the route layer, not inside
`RuleStore` itself — `ragcore.rules` core has no tenant concept baked
into its query methods (by design: it's a generic library, tenancy is
this server's own concern, the same split `ragcore.tenancy.TenantScopedCorpus`
already draws for the retrieval side).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.auth import get_tenant_id
from app.corpus_provider import get_scoped_corpus
from app.quota import enforce_quota
from app.rate_limit import enforce_rate_limit
from app.rules_models import (
    EditRuleRequest,
    ExtractRulesResponse,
    MatchRulesRequest,
    MatchRulesResponse,
    ReviewActionRequest,
    RuleResponse,
)
from app.rules_provider import (
    get_domain_pack,
    get_rule_extraction_pipeline,
    get_rule_store,
)
from app.usage import record_usage
from ragcore.rules.base import RuleCandidateDetector, RuleExtractor, RuleVerifier
from ragcore.rules.domain import DomainPack
from ragcore.rules.pipeline import build_rules
from ragcore.rules.review import ReviewQueue
from ragcore.rules.serving import match_with_provenance
from ragcore.rules.sqlite_store import SqliteRuleStore
from ragcore.rules.types import Effect, Remedy, Rule

router = APIRouter(
    prefix="/rules", tags=["rules"],
    dependencies=[Depends(enforce_rate_limit), Depends(enforce_quota), Depends(record_usage)],
)


def _visible_to_tenant(rule: Rule, tenant_id: str | None) -> bool:
    if rule.tenant_scope == "global":
        return True
    return rule.tenant_id == tenant_id


@router.get("", response_model=list[RuleResponse])
def list_rules(
    domain: str,
    book_id: str | None = None,
    store: SqliteRuleStore = Depends(get_rule_store),  # noqa: B008
    tenant_id: str | None = Depends(get_tenant_id),
) -> list[RuleResponse]:
    rules = store.rules_for_book(book_id) if book_id is not None else store.rules_for_domain(domain)
    rules = [r for r in rules if r.domain == domain and _visible_to_tenant(r, tenant_id)]
    return [RuleResponse.from_rule(r) for r in rules]


@router.post("/match", response_model=MatchRulesResponse)
def match_rules_route(
    req: MatchRulesRequest,
    store: SqliteRuleStore = Depends(get_rule_store),  # noqa: B008
    tenant_id: str | None = Depends(get_tenant_id),
) -> MatchRulesResponse:
    pool = (
        store.rules_for_book(req.book_id) if req.book_id is not None
        else store.rules_for_domain(req.domain)
    )
    pool = [r for r in pool if r.domain == req.domain and _visible_to_tenant(r, tenant_id)]
    result = match_with_provenance(
        pool, req.facts, store=store, include_unverified=req.include_unverified
    )
    return MatchRulesResponse.from_result(result)


def _get_owned_rule(store: SqliteRuleStore, rule_id: str, tenant_id: str | None) -> Rule:
    rule = store.get(rule_id)
    if rule is None or not _visible_to_tenant(rule, tenant_id):
        raise HTTPException(status_code=404, detail=f"no rule found with rule_id {rule_id!r}")
    return rule


@router.post("/{rule_id}/approve", response_model=RuleResponse)
def approve_rule(
    rule_id: str,
    req: ReviewActionRequest,
    store: SqliteRuleStore = Depends(get_rule_store),  # noqa: B008
    tenant_id: str | None = Depends(get_tenant_id),
) -> RuleResponse:
    _get_owned_rule(store, rule_id, tenant_id)
    rule = ReviewQueue(store).approve(rule_id, reviewer=req.reviewer, reason=req.reason)
    return RuleResponse.from_rule(rule)


@router.post("/{rule_id}/reject", response_model=RuleResponse)
def reject_rule(
    rule_id: str,
    req: ReviewActionRequest,
    store: SqliteRuleStore = Depends(get_rule_store),  # noqa: B008
    tenant_id: str | None = Depends(get_tenant_id),
) -> RuleResponse:
    _get_owned_rule(store, rule_id, tenant_id)
    if not req.reason.strip():
        raise HTTPException(status_code=400, detail="rejecting a rule requires a non-empty reason")
    rule = ReviewQueue(store).reject(rule_id, reviewer=req.reviewer, reason=req.reason)
    return RuleResponse.from_rule(rule)


@router.post("/{rule_id}/edit", response_model=RuleResponse)
def edit_rule(
    rule_id: str,
    req: EditRuleRequest,
    store: SqliteRuleStore = Depends(get_rule_store),  # noqa: B008
    tenant_id: str | None = Depends(get_tenant_id),
) -> RuleResponse:
    _get_owned_rule(store, rule_id, tenant_id)
    if not req.reason.strip():
        raise HTTPException(status_code=400, detail="editing a rule requires a non-empty reason")

    overrides: dict[str, object] = {}
    if req.effect is not None:
        overrides["effect"] = Effect.model_validate(req.effect)
    if req.confidence is not None:
        overrides["confidence"] = req.confidence
    if req.remedy is not None:
        overrides["remedy"] = Remedy.model_validate(req.remedy)

    rule = ReviewQueue(store).edit(rule_id, reviewer=req.reviewer, reason=req.reason, **overrides)
    return RuleResponse.from_rule(rule)


@router.post("/extract", response_model=ExtractRulesResponse)
def extract_rules(
    domain: str,
    source_id: str | None = None,
    corpus=Depends(get_scoped_corpus),  # noqa: B008
    store: SqliteRuleStore = Depends(get_rule_store),  # noqa: B008
    domain_pack: DomainPack = Depends(get_domain_pack),  # noqa: B008
    pipeline: tuple[RuleCandidateDetector, RuleExtractor, RuleVerifier] = Depends(get_rule_extraction_pipeline),
) -> ExtractRulesResponse:
    detector, extractor, verifier = pipeline
    source_ids = [source_id] if source_id is not None else None
    report = build_rules(corpus, store, detector, extractor, verifier, domain_pack, source_ids=source_ids)
    return ExtractRulesResponse(
        chapters_processed=report.chapters_processed,
        chapters_skipped=report.chapters_skipped,
        candidates_detected=report.candidates_detected,
        rules_extracted=report.rules_extracted,
        rules_stored=report.rules_stored,
        rules_rejected_unsupported=report.rules_rejected_unsupported,
        errors=list(report.errors),
    )
