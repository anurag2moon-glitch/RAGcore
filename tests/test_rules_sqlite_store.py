"""SqliteRuleStore: add_rules idempotency, get/rules_for_book/
rules_for_domain, the supersede immutable-version-on-edit discipline
(the old row's content is never touched, only a bookkeeping flag),
per-chapter checkpointing, and conflict storage — Phase 10 §1/§3/§4.
"""

from __future__ import annotations

import pytest

from ragcore.errors import RagcoreError
from ragcore.rules.consistency import find_duplicates
from ragcore.rules.sqlite_store import SqliteRuleStore
from ragcore.rules.types import Effect, FactPredicate, Rule, rule_content_hash


def _rule(
    *, book_id: str = "book1", span: str = "Mars in house 8", version: int = 1,
    supersedes: str | None = None,
) -> Rule:
    cond = FactPredicate(fact="planet.mars.house", operator="eq", value=8)
    effect = Effect(category="health", polarity="negative")
    rule_id = rule_content_hash(domain="test", book_id=book_id, verbatim_span=span, conditions=cond, effect=effect)
    return Rule(
        rule_id=rule_id,
        domain="test",
        pack_id="p",
        pack_version="1",
        book_id=book_id,
        page_start=1,
        chunk_ids=("c:0",),
        verbatim_span=span,
        language="en",
        normalized_statement="n",
        conditions=cond,
        effect=effect,
        rule_type="HARD",
        confidence=0.9,
        extraction_model="m",
        extraction_version="v1",
        created_at="now",
        version=version,
        supersedes=supersedes,
    )


# -- add_rules / get -----------------------------------------------------


def test_add_then_get_round_trips_the_full_rule() -> None:
    store = SqliteRuleStore(":memory:")
    rule = _rule()

    store.add_rules([rule])

    fetched = store.get(rule.rule_id)
    assert fetched is not None
    assert fetched.rule_id == rule.rule_id
    assert fetched.conditions == rule.conditions
    assert fetched.effect == rule.effect


def test_get_an_unknown_rule_id_returns_none() -> None:
    store = SqliteRuleStore(":memory:")
    assert store.get("rule:nonexistent") is None


def test_add_rules_is_idempotent_by_content_addressed_id() -> None:
    store = SqliteRuleStore(":memory:")
    rule = _rule()
    store.add_rules([rule])
    store.add_rules([rule])  # re-add identical content - must not raise or duplicate

    assert len(store.rules_for_book("book1")) == 1


def test_add_rules_with_an_empty_sequence_is_a_no_op() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([])
    assert store.rules_for_book("book1") == []


# -- rules_for_book / rules_for_domain -----------------------------------


def test_rules_for_book_only_returns_that_books_rules() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule(book_id="book1"), _rule(book_id="book2", span="different text")])

    assert [r.book_id for r in store.rules_for_book("book1")] == ["book1"]


def test_rules_for_domain_returns_every_rule_in_that_domain() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule(book_id="book1"), _rule(book_id="book2", span="different text")])

    assert len(store.rules_for_domain("test")) == 2


# -- supersede: immutable-version-on-edit --------------------------------


def test_supersede_writes_the_new_rule() -> None:
    store = SqliteRuleStore(":memory:")
    old = _rule()
    store.add_rules([old])
    new = _rule(span="Mars in house 8 (edited)", version=2, supersedes=old.rule_id)

    store.supersede(old.rule_id, new)

    assert store.get(new.rule_id) is not None


def test_supersede_never_mutates_the_old_rules_content() -> None:
    """The one and only mutation supersede() applies to an existing
    row is the is_latest bookkeeping flag - the old row's own payload
    (everything get() returns) must be byte-identical before and
    after."""
    store = SqliteRuleStore(":memory:")
    old = _rule()
    store.add_rules([old])
    old_before = store.get(old.rule_id)

    new = _rule(span="Mars in house 8 (edited)", version=2, supersedes=old.rule_id)
    store.supersede(old.rule_id, new)

    old_after = store.get(old.rule_id)
    assert old_after == old_before


def test_rules_for_book_latest_only_excludes_a_superseded_rule() -> None:
    store = SqliteRuleStore(":memory:")
    old = _rule()
    store.add_rules([old])
    new = _rule(span="Mars in house 8 (edited)", version=2, supersedes=old.rule_id)
    store.supersede(old.rule_id, new)

    latest = store.rules_for_book("book1")

    assert [r.rule_id for r in latest] == [new.rule_id]


def test_rules_for_book_latest_only_false_includes_every_version() -> None:
    store = SqliteRuleStore(":memory:")
    old = _rule()
    store.add_rules([old])
    new = _rule(span="Mars in house 8 (edited)", version=2, supersedes=old.rule_id)
    store.supersede(old.rule_id, new)

    everything = store.rules_for_book("book1", latest_only=False)

    assert {r.rule_id for r in everything} == {old.rule_id, new.rule_id}


def test_supersede_rejects_a_mismatched_supersedes_field() -> None:
    store = SqliteRuleStore(":memory:")
    old = _rule()
    store.add_rules([old])
    mismatched = _rule(span="completely different", version=2, supersedes="rule:wrong")

    with pytest.raises(RagcoreError, match="must equal"):
        store.supersede(old.rule_id, mismatched)


def test_supersede_rejects_an_unknown_old_rule_id() -> None:
    store = SqliteRuleStore(":memory:")
    new = _rule(version=2, supersedes="rule:nonexistent")

    with pytest.raises(RagcoreError, match="no rule found"):
        store.supersede("rule:nonexistent", new)


# -- checkpointing --------------------------------------------------------


def test_checkpoint_is_none_before_any_run() -> None:
    store = SqliteRuleStore(":memory:")
    assert store.checkpoint("book1", "ch1") is None


def test_mark_chapter_done_then_checkpoint_returns_the_fingerprint() -> None:
    store = SqliteRuleStore(":memory:")
    store.mark_chapter_done(source_id="book1", context_group_id="ch1", fingerprint="fp1")
    assert store.checkpoint("book1", "ch1") == "fp1"


def test_mark_chapter_failed_does_not_set_a_checkpoint() -> None:
    store = SqliteRuleStore(":memory:")
    store.mark_chapter_failed(source_id="book1", context_group_id="ch1", fingerprint="fp1", error="boom")
    assert store.checkpoint("book1", "ch1") is None


def test_a_failed_chapter_can_later_succeed_and_checkpoint() -> None:
    store = SqliteRuleStore(":memory:")
    store.mark_chapter_failed(source_id="book1", context_group_id="ch1", fingerprint="fp1", error="boom")
    store.mark_chapter_done(source_id="book1", context_group_id="ch1", fingerprint="fp1")
    assert store.checkpoint("book1", "ch1") == "fp1"


# -- conflicts --------------------------------------------------------------


def test_put_conflicts_then_conflicts_for_rule_finds_it_from_either_side() -> None:
    store = SqliteRuleStore(":memory:")
    r1, r2 = _rule(book_id="b1"), _rule(book_id="b2")
    store.add_rules([r1, r2])
    conflicts = find_duplicates([r1, r2])

    store.put_conflicts(conflicts)

    assert [c.conflict_id for c in store.conflicts_for_rule(r1.rule_id)] == [conflicts[0].conflict_id]
    assert [c.conflict_id for c in store.conflicts_for_rule(r2.rule_id)] == [conflicts[0].conflict_id]


def test_conflicts_for_an_untouched_rule_is_empty() -> None:
    store = SqliteRuleStore(":memory:")
    assert store.conflicts_for_rule("rule:nonexistent") == []


def test_put_conflicts_is_idempotent_by_content_addressed_id() -> None:
    store = SqliteRuleStore(":memory:")
    r1, r2 = _rule(book_id="b1"), _rule(book_id="b2")
    store.add_rules([r1, r2])
    conflicts = find_duplicates([r1, r2])

    store.put_conflicts(conflicts)
    store.put_conflicts(conflicts)  # re-insert identical conflicts

    assert len(store.conflicts_for_rule(r1.rule_id)) == 1


def test_put_conflicts_with_an_empty_sequence_is_a_no_op() -> None:
    store = SqliteRuleStore(":memory:")
    store.put_conflicts([])
    assert store.conflicts_for_rule("anything") == []


# -- review workflow (direct store-level checks; ReviewQueue's own test
# suite covers the orchestration-level behavior built on top of these) --


def test_set_review_status_rejects_an_unsupported_target_status() -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule()])
    rule_id = store.rules_for_book("book1")[0].rule_id

    with pytest.raises(RagcoreError, match="verified.*rejected"):
        store.set_review_status(rule_id, "auto", reviewer="x", reason="y")


def test_set_review_status_on_an_unknown_rule_raises() -> None:
    store = SqliteRuleStore(":memory:")
    with pytest.raises(RagcoreError, match="no rule found"):
        store.set_review_status("rule:nonexistent", "verified", reviewer="x", reason="y")


def test_record_edit_is_atomic_with_the_audit_entry() -> None:
    """A record_edit call either writes both the new version and its
    audit entry, or neither - checked here by forcing supersede's own
    validation to fail partway through and confirming no audit entry
    was left behind."""
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule()])
    rule_id = store.rules_for_book("book1")[0].rule_id
    bad_new_rule = _rule(span="different", version=2, supersedes="wrong_id")

    with pytest.raises(RagcoreError):
        store.record_edit(rule_id, bad_new_rule, reviewer="x", reason="y")

    assert store.audit_log_for_rule(rule_id) == []


# -- backup/restore --------------------------------------------------------


def test_backup_to_produces_a_restorable_copy(tmp_path) -> None:
    store = SqliteRuleStore(":memory:")
    store.add_rules([_rule()])

    backup_path = tmp_path / "backup.sqlite3"
    store.backup_to(backup_path)

    restored = SqliteRuleStore(backup_path)
    assert len(restored.rules_for_book("book1")) == 1
