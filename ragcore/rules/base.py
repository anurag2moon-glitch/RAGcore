"""Protocols for the rule extraction pipeline (Phase 10 §2) and the
rule store (§1/§4). Mirrors the shape every other extraction layer in
this codebase uses (`ragcore.graph.base`, `ragcore.concepts.base`,
`ragcore.claims.base`) — a detector/extractor/verifier each a thin
Protocol, a checkpointed store — but the store's own write discipline
is new: immutable-version-on-edit (§4), not merge-not-replace
(docs/adr/0011-graph-merge-not-replace.md) — see `RuleStore.add_rules`'s
own docstring for why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from ragcore.rules.domain import DomainPack
from ragcore.rules.types import Rule
from ragcore.types import Chunk


@dataclass(frozen=True)
class CandidateSpan:
    """One candidate rule-like span found in a chunk — Phase 10 §2
    pass 1's output, pass 2's input."""

    chunk_id: str
    source_id: str
    text: str
    """The candidate verbatim span itself — not necessarily the whole
    chunk's text; a chunk discussing several unrelated topics may
    yield zero, one, or several candidate spans."""


@dataclass(frozen=True)
class VerificationResult:
    """Phase 10 §2 pass 3's output: whether a candidate rule's
    `verbatim_span` actually supports every one of its `conditions`
    and its `effect` — "Reject anything unsupported" is this layer's
    own job, not the pipeline's; the pipeline just acts on the
    verdict."""

    rule_id: str
    supported: bool
    reason: str
    """Always populated, supported or not — a positive verification
    should still record *why*, for audit/debugging, not just a bare
    boolean."""


@runtime_checkable
class RuleCandidateDetector(Protocol):
    """Pass 1: a cheap filter — does this chunk contain candidate
    rule-like statements at all — before the more expensive structured
    extraction pass runs on it. Takes `domain_pack` (unlike a generic
    "does this look like a rule" filter would need to) because what
    counts as a candidate is itself domain-specific — a Lal Kitab
    detector should be looking for planet/house/effect language, not
    an domain-blind notion of "rule-like"."""

    model: str

    def detect(self, chunks: Sequence[Chunk], *, domain_pack: DomainPack) -> list[CandidateSpan]: ...


@runtime_checkable
class RuleExtractor(Protocol):
    """Pass 2: structured extraction of zero or more `Rule`s from a
    batch of candidate spans, using `domain_pack` for vocabulary/
    ontology/prompt guidance. `rule_id` on every returned `Rule` must
    be `ragcore.rules.types.rule_content_hash(...)` — content-addressed,
    exactly like every other extraction layer in this codebase — so a
    re-run over unchanged text reproduces the same ids."""

    model: str

    def extract(
        self, candidates: Sequence[CandidateSpan], *, domain_pack: DomainPack
    ) -> list[Rule]: ...


@runtime_checkable
class RuleVerifier(Protocol):
    """Pass 3: independent verification — never the same call/model
    state as the extractor that proposed a rule, so it can't simply
    agree with its own prior output."""

    model: str

    def verify(self, rules: Sequence[Rule], *, domain_pack: DomainPack) -> list[VerificationResult]: ...


@runtime_checkable
class RuleStore(Protocol):
    """Persists `Rule`s with the immutable-version-on-edit discipline
    (Phase 10 §4): a `rule_id` once written is **never** updated in
    place — not even a correction. An edit (`ragcore.rules.review`)
    writes a brand-new `Rule` (new content, new content-addressed
    `rule_id`, `version = old.version + 1`, `supersedes = old.rule_id`)
    and the store's only mutation of the *old* row is a bookkeeping
    "no longer latest" flag, never its content. `add_rules` itself is
    idempotent by `rule_id` (re-extracting unchanged text is a no-op,
    the same content-addressing discipline `docs/adr/0010` already
    established), not a merge — a rule, unlike an entity or concept,
    is never incrementally accumulated from multiple extraction calls.
    """

    def add_rules(self, rules: Sequence[Rule]) -> None: ...

    def get(self, rule_id: str) -> Rule | None: ...

    def rules_for_book(self, book_id: str, *, latest_only: bool = True) -> list[Rule]: ...

    def rules_for_domain(self, domain: str, *, latest_only: bool = True) -> list[Rule]: ...

    def supersede(self, old_rule_id: str, new_rule: Rule) -> None:
        """Writes `new_rule` and flips `old_rule_id`'s `is_latest`
        bookkeeping flag — the one and only mutation this store ever
        applies to an existing row, and never to the row's own rule
        content. `new_rule.supersedes` must equal `old_rule_id`."""
        ...

    def checkpoint(self, source_id: str, context_group_id: str) -> str | None: ...

    def mark_chapter_done(
        self, *, source_id: str, context_group_id: str, fingerprint: str
    ) -> None: ...

    def mark_chapter_failed(
        self, *, source_id: str, context_group_id: str, fingerprint: str, error: str
    ) -> None: ...
