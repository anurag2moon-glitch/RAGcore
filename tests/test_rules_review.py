"""ReviewQueue: list_pending ordering/filtering, approve/reject's
in-place review_status mutation (an explicit exception to immutable-
version-on-edit, since review_status is workflow metadata not rule
content), edit's genuine new-version creation, and the audit trail
reachable from either side of an edit — Phase 10 §4.
"""

from __future__ import annotations

import pytest

from ragcore.errors import RagcoreError
from ragcore.rules.review import ReviewQueue
from ragcore.rules.sqlite_store import SqliteRuleStore
from ragcore.rules.types import Effect, FactPredicate, Remedy, Rule


def _rule(
    rule_id: str, *, confidence: float = 0.5, domain: str = "test", book_id: str = "b1",
) -> Rule:
    return Rule(
        rule_id=rule_id, domain=domain, pack_id="p", pack_version="1", book_id=book_id,
        page_start=1, chunk_ids=("c:0",), verbatim_span=f"span {rule_id}", language="en",
        normalized_statement="n", conditions=FactPredicate(fact="x", operator="eq", value=8),
        effect=Effect(category="health", polarity="negative"), rule_type="HARD",
        confidence=confidence, extraction_model="m", extraction_version="v1", created_at="now",
    )


# -- list_pending -----------------------------------------------------------


def test_list_pending_returns_auto_status_rules_lowest_confidence_first() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1", confidence=0.9), _rule("r2", confidence=0.2), _rule("r3", confidence=0.5)])
    queue = ReviewQueue(store)

    pending = queue.list_pending(domain="test")

    assert [r.rule_id for r in pending] == ["r2", "r3", "r1"]


def test_list_pending_excludes_already_reviewed_rules() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1"), _rule("r2")])
    queue = ReviewQueue(store)
    queue.approve("r1", reviewer="alice")

    pending = queue.list_pending(domain="test")

    assert [r.rule_id for r in pending] == ["r2"]


def test_list_pending_filters_by_confidence_bounds() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1", confidence=0.1), _rule("r2", confidence=0.5), _rule("r3", confidence=0.9)])
    queue = ReviewQueue(store)

    pending = queue.list_pending(domain="test", min_confidence=0.3, max_confidence=0.7)

    assert [r.rule_id for r in pending] == ["r2"]


def test_list_pending_respects_limit() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule(f"r{i}", confidence=i / 10) for i in range(5)])
    queue = ReviewQueue(store)

    pending = queue.list_pending(domain="test", limit=2)

    assert len(pending) == 2


def test_list_pending_scoped_to_a_book() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1", book_id="book1"), _rule("r2", book_id="book2")])
    queue = ReviewQueue(store)

    pending = queue.list_pending(domain="test", book_id="book1")

    assert [r.rule_id for r in pending] == ["r1"]


# -- approve / reject --------------------------------------------------------


def test_approve_mutates_review_status_in_place_not_a_new_version() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1")])
    queue = ReviewQueue(store)

    approved = queue.approve("r1", reviewer="alice", reason="checks out")

    assert approved.rule_id == "r1"  # same identity - not superseded
    assert approved.review_status == "verified"
    assert approved.version == 1
    assert store.get("r1").review_status == "verified"


def test_reject_requires_a_non_empty_reason() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1")])
    queue = ReviewQueue(store)

    with pytest.raises(ValueError, match="reason"):
        queue.reject("r1", reviewer="alice", reason="")


def test_reject_sets_review_status_rejected() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1")])
    queue = ReviewQueue(store)

    rejected = queue.reject("r1", reviewer="alice", reason="unsupported")

    assert rejected.review_status == "rejected"
    assert rejected.rule_id == "r1"


def test_approve_appends_an_audit_entry() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1")])
    queue = ReviewQueue(store)

    queue.approve("r1", reviewer="alice", reason="looks right")

    entries = queue.audit_log("r1")
    assert len(entries) == 1
    assert entries[0].action == "approved"
    assert entries[0].reviewer == "alice"
    assert entries[0].reason == "looks right"
    assert entries[0].new_rule_id is None


# -- edit ---------------------------------------------------------------


def test_edit_creates_a_genuinely_new_version_not_a_mutation() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1", confidence=0.4)])
    queue = ReviewQueue(store)

    new_rule = queue.edit("r1", reviewer="carol", reason="confidence too low", confidence=0.95)

    assert new_rule.rule_id != "r1"
    assert new_rule.version == 2
    assert new_rule.supersedes == "r1"
    assert new_rule.confidence == 0.95
    assert new_rule.review_status == "verified"
    # the old row's own content is untouched - same invariant the store's
    # own supersede() test already proves, re-checked at the review layer
    old = store.get("r1")
    assert old.confidence == 0.4
    assert old.review_status == "auto"


def test_edit_can_replace_a_nested_field() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1")])
    queue = ReviewQueue(store)

    new_remedy = Remedy(action="donate", object="jaggery", day="Tuesday")
    new_rule = queue.edit("r1", reviewer="carol", reason="add missing remedy", remedy=new_remedy)

    assert new_rule.remedy == new_remedy


def test_edit_requires_a_non_empty_reason() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1")])
    queue = ReviewQueue(store)

    with pytest.raises(ValueError, match="reason"):
        queue.edit("r1", reviewer="carol", reason="", confidence=0.95)


def test_edit_of_an_unknown_rule_raises() -> None:
    store = SqliteRuleStore(":memory:")
    queue = ReviewQueue(store)

    with pytest.raises(RagcoreError, match="no rule found"):
        queue.edit("rule:nonexistent", reviewer="carol", reason="x", confidence=0.5)


def test_edit_replaying_the_same_edit_produces_the_same_new_rule_id() -> None:
    """Deterministic, not random - a retried edit call (e.g. after a
    transient failure) is safe because it reproduces the exact same
    new_rule_id rather than minting a second, different version."""
    store1 = SqliteRuleStore(":memory:")
    store1.add_rules([_rule("r1", confidence=0.4)])
    store2 = SqliteRuleStore(":memory:")
    store2.add_rules([_rule("r1", confidence=0.4)])

    new1 = ReviewQueue(store1).edit("r1", reviewer="carol", reason="bump", confidence=0.95)
    new2 = ReviewQueue(store2).edit("r1", reviewer="carol", reason="bump", confidence=0.95)

    assert new1.rule_id == new2.rule_id


def test_edit_is_reachable_in_the_new_rules_latest_only_listing() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1", book_id="book1")])
    queue = ReviewQueue(store)

    new_rule = queue.edit("r1", reviewer="carol", reason="bump", confidence=0.95)

    latest = store.rules_for_book("book1")
    assert [r.rule_id for r in latest] == [new_rule.rule_id]


# -- audit log: reachable from either side of an edit ------------------


def test_audit_log_for_an_edit_is_reachable_from_both_old_and_new_rule_id() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1")])
    queue = ReviewQueue(store)

    new_rule = queue.edit("r1", reviewer="carol", reason="bump", confidence=0.95)

    from_old = queue.audit_log("r1")
    from_new = queue.audit_log(new_rule.rule_id)
    assert len(from_old) == 1
    assert from_old[0].action == "edited"
    assert from_old[0].new_rule_id == new_rule.rule_id
    assert from_old == from_new


def test_audit_log_accumulates_multiple_actions_on_the_same_rule_id_in_order() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1")])
    queue = ReviewQueue(store)

    queue.approve("r1", reviewer="dave", reason="first pass")

    # reject() after approve() is a legal (if unusual) second transition on
    # the same still-latest rule_id - both land in the same rule_id's log.
    queue.reject("r1", reviewer="dave", reason="actually, reconsidered")

    history = queue.audit_log("r1")
    assert [h.action for h in history] == ["approved", "rejected"]


def test_audit_log_lookup_is_single_hop_not_transitive_across_the_whole_lineage() -> None:
    """audit_log_for_rule(rule_id) finds entries naming rule_id on
    either side of ONE transition (the rule acted upon, or - for an
    edit - the resulting new version). An action taken on a LATER
    version two edits downstream is not folded into an earlier
    version's own log - this is a deliberate, stated scope limit, not
    an oversight; walking the full supersede chain would need either a
    shared lineage id or a graph walk, neither of which this store
    implements."""
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule("r1", confidence=0.4)])
    queue = ReviewQueue(store)

    new_rule = queue.edit("r1", reviewer="carol", reason="bump", confidence=0.95)
    queue.approve(new_rule.rule_id, reviewer="dave", reason="now looks good")

    assert [h.action for h in queue.audit_log("r1")] == ["edited"]
    assert [h.action for h in queue.audit_log(new_rule.rule_id)] == ["edited", "approved"]
