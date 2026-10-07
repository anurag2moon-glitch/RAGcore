"""Rule serving (Phase 10 §5): "Query API by structured facts: given a
chart fact set, return all matching rules with provenance, grouped by
conflicts and precedence."

`match_with_provenance` is the one new piece here — condition matching
itself is already `ragcore.rules.evaluator.match_rules` (pure, no LLM,
already built for §1/§5's "Evaluator library"). What this module adds
is grouping matched rules by the conflicts already on file for them
(`RuleStore.conflicts_for_rule`) and surfacing which ones are
overridden by another matched rule via an already-resolved conflict.

**This reads conflicts the store already has — it does not detect new
ones.** Conflict detection (`ragcore.rules.consistency.detect_conflicts`
+ `RuleStore.put_conflicts`) is a separate, batch maintenance pass over
the whole rule base, the same relationship `ragcore.concepts.communities`
has to retrieval: run periodically or after a build_rules()/review
pass, not on every query. A rule base that's never had conflict
detection run over it still serves correct (if conflict-unaware)
results here — `effective` just equals `matched` in full, which is a
safe degradation, not a silent correctness gap, since nothing here
ever fabricates a conflict that wasn't actually detected and stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ragcore.rules.evaluator import match_rules

if TYPE_CHECKING:
    from ragcore.rules.base import RuleStore
    from ragcore.rules.consistency import RuleConflict
    from ragcore.rules.types import Rule

FactSet = dict[str, Any]


@dataclass(frozen=True)
class MatchedRule:
    rule: Rule
    conflicts: tuple[RuleConflict, ...]
    """Every stored conflict naming this rule where the *other* side
    also matched this same fact set — a conflict with a rule that
    didn't match isn't relevant to this query's answer."""
    is_overridden: bool
    """True if some other rule that also matched has a resolved
    conflict naming it as the winner over this one."""
    overrides: tuple[str, ...]
    """rule_ids this rule itself wins over, per a resolved conflict."""


@dataclass(frozen=True)
class RuleMatchResult:
    matched: tuple[MatchedRule, ...]
    """Every rule whose conditions held against the given facts, with
    its conflict context — the full picture, including rules an
    EXCEPTION overrides, kept visible rather than discarded, so a
    consumer can show *why* a general rule didn't end up applying."""
    effective: tuple[Rule, ...]
    """matched rules that are NOT overridden by another matched rule —
    the precedence-resolved answer a deterministic engine should
    actually act on."""


def match_with_provenance(
    rules: list[Rule],
    facts: FactSet,
    *,
    store: RuleStore,
    include_unverified: bool = False,
) -> RuleMatchResult:
    """`rules` is the candidate pool (typically `store.rules_for_domain(domain)`
    or `store.rules_for_book(book_id)`) — passed explicitly rather than
    looked up by this function so a caller controls domain/book scope
    and `latest_only` the same way `ragcore.rules.evaluator.match_rules`
    always has. `include_unverified=False` (Phase 10 §4's "Only
    verified rules are served to production consumers by default") —
    an "auto" (unreviewed) or "rejected" rule is excluded unless the
    caller opts in."""
    if not include_unverified:
        rules = [r for r in rules if r.review_status == "verified"]

    candidates = match_rules(rules, facts)
    rules_by_id = {r.rule_id: r for r in candidates}
    matched_ids = set(rules_by_id)

    conflicts_by_rule: dict[str, list[RuleConflict]] = {rid: [] for rid in matched_ids}
    overridden: set[str] = set()
    overrides_by_rule: dict[str, list[str]] = {rid: [] for rid in matched_ids}

    for rule_id in matched_ids:
        for conflict in store.conflicts_for_rule(rule_id):
            other = conflict.rule_ids[1] if conflict.rule_ids[0] == rule_id else conflict.rule_ids[0]
            if other not in matched_ids:
                continue
            conflicts_by_rule[rule_id].append(conflict)
            if conflict.winning_rule_id == other:
                overridden.add(rule_id)
            elif conflict.winning_rule_id == rule_id:
                overrides_by_rule[rule_id].append(other)

    matched = tuple(
        MatchedRule(
            rule=rules_by_id[rid],
            conflicts=tuple(conflicts_by_rule[rid]),
            is_overridden=rid in overridden,
            overrides=tuple(overrides_by_rule[rid]),
        )
        for rid in sorted(matched_ids)
    )
    effective = tuple(m.rule for m in matched if not m.is_overridden)
    return RuleMatchResult(matched=matched, effective=effective)
