"""Data types for the claim extraction and linking layer.

`SourceClaim` — not `Claim` — deliberately avoids colliding with
`ragcore.verification.types.ClaimCheck.claim`, a bare `str` naming an atomic
assertion *within a generated answer*. A `SourceClaim` is a different thing:
an atomic factual assertion made *by a source document*. The two are likely
to be imported side by side by a downstream generation layer, so the names
must not collide even informally.

This layer is storage-only: it records what each source claims and how
those claims relate to each other (agreement, conflict, or one claim
extending/refining another). It does not decide which claim is "right," does
not rank claims, and is never read directly by `ClaudeGenerator` or
`VerifyingGenerator` in this codebase — see `ragcore/claims/__init__.py` and
CLAUDE.md for the full rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ClaimConfidence = Literal["firm", "hedged"]
"""Whether the author asserted a claim directly ("firm") or hedged it
himself in that exact passage ("hedged" — "perhaps", "I suspect", "it seems
to me"). Judged from that passage's own wording ONLY — never from whether
the claim is consistent with the author's other claims elsewhere. Keeping
these two judgments independent is what lets a downstream layer later
distinguish "the author himself was unsure here" from "these two passages
are in tension" — collapsing them into one signal would destroy that
distinction."""

ClaimRelationType = Literal["agrees", "conflicts", "extends"]
""""agrees"/"conflicts" are symmetric (order-independent); "extends" is
directional — the source claim refines or develops the target claim
further. See `_relation_id` in `ragcore.claims.claude` for the
canonicalization each type gets."""


@dataclass(frozen=True)
class SourceClaim:
    """One atomic factual/philosophical assertion, extracted from one chunk,
    anchored to one or more of the concept layer's already-resolved
    `concept_id`s (see `ragcore.concepts.types.Concept`) — "claims about the
    same concept" is a plain join on `concept_ids`, not a separate
    subject-clustering mechanism."""

    claim_id: str
    """Content-addressed: f"claim:{sha256(source_id|chunk_id|normalized_text)[:16]}".
    Deliberately NOT hashed over text alone — that would collapse two
    different sources independently asserting the same thing into a single
    row, destroying the very agreement signal this layer exists to
    capture."""

    concept_ids: tuple[str, ...]
    source_id: str
    chunk_id: str

    text: str
    """A verbatim span of the source chunk, not a paraphrase — same
    discipline as `ClaimCheck.claim` in `ragcore.verification.types`."""

    author_confidence: ClaimConfidence

    model: str
    prompt_version: str
    created_at: str

    vector: tuple[float, ...] | None = None
    """The claim text's own embedding, persisted at extraction time (never
    recomputed at link time) — same discipline as `Concept.vector` /
    `CommunitySummary.vector`. Powers the embedding-shortlist candidate
    bounding in `ragcore.claims.linking.link_claims`."""

    tenant_id: str | None = None
    """Same meaning as `ragcore.types.ChunkMeta.tenant_id` — taken
    directly from the source chunk this claim was extracted from
    (`ClaudeClaimExtractor.extract`), since claim_id is already scoped
    by `(source_id, chunk_id)` and every chunk of one source shares one
    tenant_id by construction."""

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClaimRelation:
    """One agreement/conflict/extension found between two `SourceClaim`s,
    scoped to whatever `ClaimStore` they both live in — which, per this
    session's product design, always means one persona's own corpus. There
    is no cross-store/cross-persona linking anywhere in this layer."""

    relation_id: str
    """Type-dependent canonicalization (the relation type is baked into the
    id on purpose — see `ragcore.claims.claude._relation_id` — otherwise a
    re-run of linking could flip an existing row's type nondeterministically):
    - "agrees"/"conflicts" (symmetric): endpoints sorted,
      f"claimrel:{type}:{a}:{b}" with a < b.
    - "extends" (directional — source refines/develops target): endpoints
      NOT sorted, f"claimrel:extends:{source}:{target}"."""

    source_claim_id: str
    target_claim_id: str
    relation_type: ClaimRelationType

    rationale: str
    """Required, not optional — never folded into relation_id."""

    model: str
    prompt_version: str
    created_at: str
