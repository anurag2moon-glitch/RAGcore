"""Claim extraction and linking layer: extracts atomic factual/philosophical
claims from already-ingested chunks, anchors each to the concept layer's
already-resolved concepts, and finds agree/conflict/extends relations
between claims anchored to the same concept.

This is a storage-only layer: it has no retriever, and nothing here is read
by ragcore.generation or ragcore.verification. It exists so a later,
domain-specific layer can decide how to speak about a persona's apparently
conflicting claims (e.g. presenting a later position as a refinement rather
than exposing "source A says X, source B says Y" to an end user) — this
package only makes that later layer *buildable*, by recording each claim's
own author_confidence ("firm" vs. "hedged", judged from that passage's own
wording) and every relation between claims, without doing any resolution or
ranking of its own.

Linking scope: every claim reachable in one ClaimStore is comparable —
there is no persona/author field anywhere in this package, because the
assumption (locked for this product) is that each persona has its own
Corpus and its own ClaimStore. Cross-persona comparison is out of scope by
construction, not by a filter.

    from ragcore.claims import (
        SourceClaim, ClaimRelation,                          # types
        ClaimStore, ClaimExtractor, ClaimLinker,              # protocols
        SqliteClaimStore, ClaudeClaimExtractor, ClaudeClaimLinker,  # defaults
        build_claims, ClaimReport, ClaimProgress,             # extraction pass
        link_claims, LinkReport,                              # linking pass
    )
"""

from ragcore.claims.base import ClaimExtractor, ClaimLinker, ClaimStore
from ragcore.claims.claude import ClaudeClaimExtractor, ClaudeClaimLinker
from ragcore.claims.linking import LinkReport, link_claims
from ragcore.claims.pipeline import ClaimProgress, ClaimReport, build_claims
from ragcore.claims.sqlite_store import SqliteClaimStore
from ragcore.claims.types import ClaimRelation, SourceClaim

__all__ = [
    "SourceClaim",
    "ClaimRelation",
    "ClaimStore",
    "ClaimExtractor",
    "ClaimLinker",
    "SqliteClaimStore",
    "ClaudeClaimExtractor",
    "ClaudeClaimLinker",
    "build_claims",
    "ClaimReport",
    "ClaimProgress",
    "link_claims",
    "LinkReport",
]
