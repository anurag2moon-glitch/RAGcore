"""Review workflow (Phase 10 §4): a review queue over whatever a
RuleStore already holds, plus the audit trail every approve/reject/edit
leaves behind.

`review_status` ("auto"/"verified"/"rejected") is deliberately **not**
part of `rule_content_hash`'s inputs (see `ragcore.rules.types`) — it's
workflow metadata about a rule, not the rule's own substance. Approving
or rejecting a rule therefore mutates `review_status` on the existing
row in place (`RuleStore.set_review_status`), a narrow, explicit
exception to the immutable-version-on-edit discipline `RuleStore.add_rules`'s
own docstring describes for rule *content* — the audit log, not version
chaining, is what makes a status transition non-destructively
traceable here.

An actual content edit (`ReviewQueue.edit`) is a different thing
entirely and **does** go through `RuleStore.supersede` exactly like
`ragcore.rules.pipeline` would for any other new version — "Edits
create new versions and never mutate in place" (Phase 10 §4), the rule
this module's `edit` is named for. Its new `rule_id` isn't
`rule_content_hash` either, deliberately: that hash is coarse-grained,
designed so the same rule re-extracted collapses to one id even across
books — exactly the wrong property for an edit, where a reviewer
correcting even a field `rule_content_hash` ignores (confidence,
remedy, rule_type, ...) must still produce a genuinely new identity.
Instead, the edited rule's full content (everything except its own
`rule_id`) is hashed, with `version`/`supersedes` already set first —
`supersedes = old.rule_id` alone guarantees the new id can never
collide with the old one, so this is deterministic (the same edit
replayed twice reproduces the same new `rule_id`, safe to retry) and
always produces a new identity, not just when a `rule_content_hash`
input happened to change.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ragcore.errors import RagcoreError

if TYPE_CHECKING:
    from ragcore.rules.base import RuleStore
    from ragcore.rules.types import Rule

ReviewAction = Literal["approved", "rejected", "edited"]


@dataclass(frozen=True)
class ReviewAuditEntry:
    entry_id: str
    rule_id: str
    """The rule acted upon. For an "edited" entry, this is the *old*
    rule_id — the one that was superseded, not the new version;
    `RuleStore.audit_log_for_rule` looks up entries by either side so
    an edit's history is reachable from either id."""
    action: ReviewAction
    reviewer: str
    reason: str
    created_at: str
    new_rule_id: str | None = None
    """Only set for "edited" — the resulting new version's rule_id."""


def _new_version_rule_id(candidate: Rule) -> str:
    payload = candidate.model_dump_json(exclude={"rule_id"})
    return f"rule:edit:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


class ReviewQueue:
    """The operational surface Phase 10 §4 describes: "Review queue
    API: list by confidence, domain, book. Approve / edit / reject
    with an audit log." A thin orchestration layer over whatever
    RuleStore it's given — holds no state of its own."""

    def __init__(self, store: RuleStore) -> None:
        self._store = store

    def list_pending(
        self,
        *,
        domain: str,
        book_id: str | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
        limit: int = 50,
    ) -> list[Rule]:
        """Rules still awaiting review (review_status == "auto"),
        lowest-confidence-first — the natural review priority, since a
        low-confidence extraction is the one most likely to need a
        human's attention."""
        rules = (
            self._store.rules_for_book(book_id) if book_id is not None
            else self._store.rules_for_domain(domain)
        )
        pending = [r for r in rules if r.domain == domain and r.review_status == "auto"]
        if min_confidence is not None:
            pending = [r for r in pending if r.confidence >= min_confidence]
        if max_confidence is not None:
            pending = [r for r in pending if r.confidence <= max_confidence]
        pending.sort(key=lambda r: r.confidence)
        return pending[:limit]

    def approve(self, rule_id: str, *, reviewer: str, reason: str = "") -> Rule:
        return self._store.set_review_status(rule_id, "verified", reviewer=reviewer, reason=reason)

    def reject(self, rule_id: str, *, reviewer: str, reason: str) -> Rule:
        if not reason.strip():
            raise ValueError("rejecting a rule requires a non-empty reason")
        return self._store.set_review_status(rule_id, "rejected", reviewer=reviewer, reason=reason)

    def edit(self, rule_id: str, *, reviewer: str, reason: str, **overrides: Any) -> Rule:
        """Constructs a new version of `rule_id` with `overrides`
        applied (any `Rule` field — e.g. `conditions=...`, `effect=...`,
        `remedy=...`), writes it via `RuleStore.supersede`, and appends
        one audit entry. `overrides` must pass real typed values for
        any nested field (a `Condition`/`Effect`/`Remedy` instance, not
        a plain dict) — the same contract `Rule.model_copy(update=...)`
        itself has."""
        if not reason.strip():
            raise ValueError("editing a rule requires a non-empty reason")
        old = self._store.get(rule_id)
        if old is None:
            raise RagcoreError(f"no rule found with rule_id {rule_id!r}")

        candidate = old.model_copy(
            update={
                **overrides,
                "version": old.version + 1,
                "supersedes": old.rule_id,
                "review_status": "verified",
            }
        )
        new_rule = candidate.model_copy(update={"rule_id": _new_version_rule_id(candidate)})

        self._store.record_edit(old.rule_id, new_rule, reviewer=reviewer, reason=reason)
        return new_rule

    def audit_log(self, rule_id: str) -> list[ReviewAuditEntry]:
        return self._store.audit_log_for_rule(rule_id)
