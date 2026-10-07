"""RagcoreClient — a thin, typed wrapper over the reference server's HTTP
API (server/app/main.py). See server/README.md for the endpoint table
this mirrors.

`http_client`, like every swappable component elsewhere in this
codebase family, is constructor-injectable — real usage passes only
`base_url` and gets a real `httpx.Client`; tests pass an `httpx.Client`
built with `transport=httpx.ASGITransport(app=...)` to exercise this
SDK directly against the FastAPI app object, in-process, with no
network and no running server.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import httpx

from ragcore_client.errors import RagcoreClientError
from ragcore_client.types import (
    AskResult,
    ExtractRulesResult,
    IngestResult,
    MatchedRuleInfo,
    MatchRulesResult,
    RetrievedChunk,
    RuleInfo,
    SourceInfo,
)


def _chunk_from_dict(d: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=d["chunk_id"],
        text=d["text"],
        score=d["score"],
        source_id=d["source_id"],
        source_path=d["source_path"],
        chapter_title=d.get("chapter_title"),
        tenant_id=d.get("tenant_id"),
        license_id=d.get("license_id"),
    )


def _rule_from_dict(d: dict[str, Any]) -> RuleInfo:
    return RuleInfo(
        rule_id=d["rule_id"], domain=d["domain"], book_id=d["book_id"], chapter=d.get("chapter"),
        page_start=d["page_start"], verbatim_span=d["verbatim_span"],
        normalized_statement=d["normalized_statement"], conditions=d["conditions"],
        effect=d["effect"], remedy=d.get("remedy"), rule_type=d["rule_type"],
        confidence=d["confidence"], review_status=d["review_status"], version=d["version"],
    )


class RagcoreClient:
    def __init__(self, base_url: str | None = None, *, http_client: httpx.Client | None = None) -> None:
        if http_client is None:
            if base_url is None:
                raise ValueError("base_url is required unless http_client is given")
            http_client = httpx.Client(base_url=base_url)
        self._client = http_client

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        resp = self._client.request(method, path, **kwargs)
        if resp.status_code >= 400:
            detail = resp.text
            try:
                detail = resp.json().get("detail", detail)
            except ValueError:
                pass
            raise RagcoreClientError(resp.status_code, detail)
        return resp

    def health(self) -> bool:
        return self._request("GET", "/health").json().get("status") == "ok"

    def ingest(self, path: str | Path) -> IngestResult:
        path = Path(path)
        with path.open("rb") as f:
            resp = self._request("POST", "/ingest", files={"file": (path.name, f)})
        body = resp.json()
        return IngestResult(
            source_ids=body["source_ids"],
            chunks_ingested=body["chunks_ingested"],
            sources_replaced=body["sources_replaced"],
            errors=[tuple(e) for e in body["errors"]],
        )

    def retrieve(
        self,
        query: str,
        *,
        k: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        resp = self._request(
            "POST",
            "/retrieve",
            json={"query": query, "k": k, "filters": filters, "min_score": min_score},
        )
        return [_chunk_from_dict(r) for r in resp.json()["results"]]

    def ask(self, query: str, *, k: int = 10, filters: dict[str, Any] | None = None) -> AskResult:
        resp = self._request("POST", "/ask", json={"query": query, "k": k, "filters": filters})
        body = resp.json()
        return AskResult(
            text=body["text"], citations=[_chunk_from_dict(c) for c in body["citations"]]
        )

    def sources(self) -> list[SourceInfo]:
        resp = self._request("GET", "/sources")
        return [
            SourceInfo(
                source_id=s["source_id"],
                path=s["path"],
                title=s["title"],
                source_type=s["source_type"],
                chunk_count=s["chunk_count"],
            )
            for s in resp.json()
        ]

    def delete_source(self, source_id: str) -> int:
        resp = self._request("DELETE", f"/sources/{source_id}")
        return resp.json()["chunks_deleted"]

    # -- Phase 10: rules (server/app/rules_routes.py) ------------------

    def list_rules(self, domain: str, *, book_id: str | None = None) -> list[RuleInfo]:
        params: dict[str, Any] = {"domain": domain}
        if book_id is not None:
            params["book_id"] = book_id
        resp = self._request("GET", "/rules", params=params)
        return [_rule_from_dict(r) for r in resp.json()]

    def match_rules(
        self,
        domain: str,
        facts: dict[str, Any],
        *,
        book_id: str | None = None,
        include_unverified: bool = False,
    ) -> MatchRulesResult:
        resp = self._request(
            "POST", "/rules/match",
            json={
                "domain": domain, "facts": facts, "book_id": book_id,
                "include_unverified": include_unverified,
            },
        )
        body = resp.json()
        return MatchRulesResult(
            matched=[
                MatchedRuleInfo(
                    rule=_rule_from_dict(m["rule"]), is_overridden=m["is_overridden"],
                    overrides=tuple(m["overrides"]),
                )
                for m in body["matched"]
            ],
            effective=[_rule_from_dict(r) for r in body["effective"]],
        )

    def approve_rule(self, rule_id: str, *, reviewer: str, reason: str = "") -> RuleInfo:
        resp = self._request(
            "POST", f"/rules/{rule_id}/approve", json={"reviewer": reviewer, "reason": reason}
        )
        return _rule_from_dict(resp.json())

    def reject_rule(self, rule_id: str, *, reviewer: str, reason: str) -> RuleInfo:
        resp = self._request(
            "POST", f"/rules/{rule_id}/reject", json={"reviewer": reviewer, "reason": reason}
        )
        return _rule_from_dict(resp.json())

    def edit_rule(
        self,
        rule_id: str,
        *,
        reviewer: str,
        reason: str,
        effect: dict[str, Any] | None = None,
        confidence: float | None = None,
        remedy: dict[str, Any] | None = None,
    ) -> RuleInfo:
        resp = self._request(
            "POST", f"/rules/{rule_id}/edit",
            json={
                "reviewer": reviewer, "reason": reason, "effect": effect,
                "confidence": confidence, "remedy": remedy,
            },
        )
        return _rule_from_dict(resp.json())

    def extract_rules(self, domain: str, *, source_id: str | None = None) -> ExtractRulesResult:
        params: dict[str, Any] = {"domain": domain}
        if source_id is not None:
            params["source_id"] = source_id
        resp = self._request("POST", "/rules/extract", params=params)
        body = resp.json()
        return ExtractRulesResult(
            chapters_processed=body["chapters_processed"], chapters_skipped=body["chapters_skipped"],
            candidates_detected=body["candidates_detected"], rules_extracted=body["rules_extracted"],
            rules_stored=body["rules_stored"],
            rules_rejected_unsupported=body["rules_rejected_unsupported"],
            errors=[tuple(e) for e in body["errors"]],
        )
