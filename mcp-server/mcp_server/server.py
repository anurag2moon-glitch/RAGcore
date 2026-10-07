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
from mcp_server.rules_provider import (
    get_domain_pack_registry,
    get_rule_extraction_pipeline,
    get_rule_store,
)
from ragcore.corpus import Corpus
from ragcore.rules.base import RuleCandidateDetector, RuleExtractor, RuleVerifier
from ragcore.rules.domain import DomainPack
from ragcore.rules.pipeline import build_rules
from ragcore.rules.review import ReviewQueue
from ragcore.rules.serving import match_with_provenance
from ragcore.rules.sqlite_store import SqliteRuleStore
from ragcore.rules.types import Effect, Remedy, Rule
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


def _rule_to_dict(rule: Rule) -> dict[str, Any]:
    return {
        "rule_id": rule.rule_id,
        "domain": rule.domain,
        "book_id": rule.book_id,
        "chapter": rule.chapter,
        "page_start": rule.page_start,
        "verbatim_span": rule.verbatim_span,
        "normalized_statement": rule.normalized_statement,
        "conditions": rule.conditions.model_dump(),
        "effect": rule.effect.model_dump(),
        "remedy": rule.remedy.model_dump() if rule.remedy else None,
        "rule_type": rule.rule_type,
        "confidence": rule.confidence,
        "review_status": rule.review_status,
        "version": rule.version,
    }


def build_server(
    *,
    corpus_provider: Callable[[], Corpus] = get_corpus,
    rule_store_provider: Callable[[], SqliteRuleStore] = get_rule_store,
    domain_pack_registry_provider: Callable[[], dict[str, DomainPack]] = get_domain_pack_registry,
    rule_extraction_pipeline_provider: Callable[
        [], tuple[RuleCandidateDetector, RuleExtractor, RuleVerifier]
    ] = get_rule_extraction_pipeline,
) -> MCPServer:
    server = MCPServer(
        "ragcore",
        description=(
            "Single-tenant access to one ragcore.Corpus and its rule base: "
            "ingest, retrieve, ask, list sources, extract/list/match/review rules."
        ),
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

    @server.tool()
    def list_rules(domain: str, book_id: str | None = None) -> list[dict[str, Any]]:
        """List rules for a domain (optionally scoped to one book)."""
        store = rule_store_provider()
        rules = store.rules_for_book(book_id) if book_id is not None else store.rules_for_domain(domain)
        return [_rule_to_dict(r) for r in rules if r.domain == domain]

    @server.tool()
    def match_rules(
        domain: str,
        facts: dict[str, Any],
        book_id: str | None = None,
        include_unverified: bool = False,
    ) -> dict[str, Any]:
        """Match a fact set against this domain's rules, grouped by
        conflict/precedence. Only verified rules are served by default
        (Phase 10 §4)."""
        store = rule_store_provider()
        pool = store.rules_for_book(book_id) if book_id is not None else store.rules_for_domain(domain)
        pool = [r for r in pool if r.domain == domain]
        result = match_with_provenance(pool, facts, store=store, include_unverified=include_unverified)
        return {
            "matched": [
                {
                    "rule": _rule_to_dict(m.rule),
                    "is_overridden": m.is_overridden,
                    "overrides": list(m.overrides),
                }
                for m in result.matched
            ],
            "effective": [_rule_to_dict(r) for r in result.effective],
        }

    @server.tool()
    def review_rule(
        rule_id: str,
        action: str,
        reviewer: str,
        reason: str = "",
        effect: dict[str, Any] | None = None,
        confidence: float | None = None,
        remedy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Approve, reject, or edit a rule (action: "approve" | "reject"
        | "edit"). Edit accepts any of effect/confidence/remedy as
        overrides and always creates a new version
        (ragcore.rules.review's immutable-version-on-edit discipline)."""
        store = rule_store_provider()
        queue = ReviewQueue(store)
        if action == "approve":
            rule = queue.approve(rule_id, reviewer=reviewer, reason=reason)
        elif action == "reject":
            rule = queue.reject(rule_id, reviewer=reviewer, reason=reason)
        elif action == "edit":
            overrides: dict[str, Any] = {}
            if effect is not None:
                overrides["effect"] = Effect.model_validate(effect)
            if confidence is not None:
                overrides["confidence"] = confidence
            if remedy is not None:
                overrides["remedy"] = Remedy.model_validate(remedy)
            rule = queue.edit(rule_id, reviewer=reviewer, reason=reason, **overrides)
        else:
            raise ValueError(f"action must be 'approve', 'reject', or 'edit', got {action!r}")
        return _rule_to_dict(rule)

    @server.tool()
    def extract_rules(domain: str, source_id: str | None = None) -> dict[str, Any]:
        """Run the 3-pass extraction pipeline (detect/extract/verify)
        over this corpus (optionally scoped to one source) for
        `domain`, storing every verified rule."""
        corpus = corpus_provider()
        store = rule_store_provider()
        registry = domain_pack_registry_provider()
        domain_pack = registry.get(domain)
        if domain_pack is None:
            raise ValueError(f"no domain pack registered for domain {domain!r} (registered: {sorted(registry)})")
        detector, extractor, verifier = rule_extraction_pipeline_provider()
        source_ids = [source_id] if source_id is not None else None
        report = build_rules(corpus, store, detector, extractor, verifier, domain_pack, source_ids=source_ids)
        return {
            "chapters_processed": report.chapters_processed,
            "chapters_skipped": report.chapters_skipped,
            "candidates_detected": report.candidates_detected,
            "rules_extracted": report.rules_extracted,
            "rules_stored": report.rules_stored,
            "rules_rejected_unsupported": report.rules_rejected_unsupported,
            "errors": list(report.errors),
            "total_tokens_used": report.total_tokens_used,
        }

    return server


server = build_server()


if __name__ == "__main__":
    server.run()
