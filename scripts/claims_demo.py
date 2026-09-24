"""Demonstrates the claim extraction and linking layer end to end:
ingest -> build_concepts -> build_claims -> link_claims -> inspecting the
resulting agree/conflict relations and author_confidence values.

No API calls: a small synthetic two-chapter passage (one persona's own
corpus) with a deliberate agreement and a deliberate conflict on the same
concept is ingested with FakeEmbedder/InMemoryStore, and every LLM-backed
component is stood in for by a small deterministic fake — the same role
scripts/concepts_demo.py's fakes play.

Usage:
    python scripts/claims_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ragcore.claims.claude import _claim_id, _relation_id
from ragcore.claims.linking import link_claims
from ragcore.claims.pipeline import build_claims
from ragcore.claims.sqlite_store import SqliteClaimStore
from ragcore.claims.types import ClaimRelation, SourceClaim
from ragcore.concepts.pipeline import build_concepts
from ragcore.concepts.sqlite_store import SqliteConceptStore
from ragcore.concepts.types import Concept
from ragcore.corpus import Corpus
from ragcore.types import Chunk
from tests.fakes import FakeEmbedder, InMemoryStore

SAMPLES_DIR = Path("samples")
BOOK_PATH = SAMPLES_DIR / "claims_demo_book.md"

BOOK_TEXT = """\
# Chapter 1: On Discipline

Discipline is a struggle each of us must wage against our own weaker
impulses, day after day, with no promise of final victory.

# Chapter 2: A Later Reflection

I have come to feel that discipline is, in the end, simply a gift some are
given and others are not — though I confess I am not entirely sure of this.
"""


class _FakeConceptExtractor:
    model = "fake-concepts"

    def extract(self, chunks: Sequence[Chunk]):
        concept = Concept(
            concept_id="concept:discipline", label="the nature of discipline",
            description="What discipline fundamentally is and how it is attained.",
            source_chunk_ids=tuple(c.chunk_id for c in chunks),
        )
        return [concept], []


class _FakeClaimExtractor:
    """Extracts one claim per chapter, tagging the second chapter's claim
    'hedged' because of its own "I confess I am not entirely sure" language
    — never because it conflicts with chapter 1."""

    model = "fake-claims"

    def extract(self, *, chunks: Sequence[Chunk], concepts):
        chunk = chunks[0]
        concept_ids = tuple(c.concept_id for c in concepts)
        if "struggle" in chunk.text:
            text, confidence = "Discipline is a struggle waged against one's own weaker impulses.", "firm"
        else:
            text, confidence = "Discipline is a gift given to some and not others.", "hedged"
        return [SourceClaim(
            claim_id=_claim_id(chunk.meta.source_id, chunk.chunk_id, text),
            concept_ids=concept_ids, source_id=chunk.meta.source_id, chunk_id=chunk.chunk_id,
            text=text, author_confidence=confidence,
            model=self.model, prompt_version="v1", created_at="now",
        )]


class _FakeClaimLinker:
    """Reports a 'conflicts' relation whenever a batch contains both the
    struggle-based and the gift-based claim."""

    model = "fake-linker"

    def link(self, claims):
        texts = {c.claim_id: c.text.lower() for c in claims}
        struggle_id = next((cid for cid, t in texts.items() if "struggle" in t), None)
        gift_id = next((cid for cid, t in texts.items() if "gift" in t), None)
        if struggle_id is None or gift_id is None:
            return []
        relation_id, source, target = _relation_id(struggle_id, gift_id, "conflicts")
        return [ClaimRelation(
            relation_id=relation_id, source_claim_id=source, target_claim_id=target,
            relation_type="conflicts",
            rationale="one claims discipline is earned through struggle, the other that it is a given gift",
            model=self.model, prompt_version="v1", created_at="now",
        )]


def main() -> None:
    SAMPLES_DIR.mkdir(exist_ok=True)
    BOOK_PATH.write_text(BOOK_TEXT)

    corpus = Corpus(InMemoryStore(dim=16), FakeEmbedder(dim=16))
    corpus.ingest(BOOK_PATH)

    concept_store = SqliteConceptStore(":memory:")
    build_concepts(corpus, concept_store, _FakeConceptExtractor())

    claim_store = SqliteClaimStore(":memory:")
    claim_report = build_claims(corpus, claim_store, _FakeClaimExtractor(), concept_store)
    print(f"=== build_claims: {claim_report.claims_extracted} claims extracted ===")

    link_report = link_claims(claim_store, _FakeClaimLinker())
    print(f"=== link_claims: {link_report.relations_found} relation(s) found ===\n")

    claims = claim_store.claims_for_concept("concept:discipline")
    print("Claims about 'concept:discipline':")
    for c in claims:
        print(f"  [{c.author_confidence:6}] {c.text!r}  (source={c.source_id[:12]}...)")

    print("\nRelations:")
    for c in claims:
        for rel in claim_store.relations_for_claim(c.claim_id):
            if rel.source_claim_id == c.claim_id:  # print each relation once
                print(f"  {rel.relation_type}: {rel.rationale}")


if __name__ == "__main__":
    main()
