"""SqliteClaimStore.remove_chunk_ids / remove_source: cascading deletes to
claim_concepts and to every relation touching a removed claim in EITHER
direction, plus checkpoint cleanup — mirroring test_concept_removal.py's
shape.
"""

from __future__ import annotations

from ragcore.claims.claude import _relation_id
from ragcore.claims.sqlite_store import SqliteClaimStore
from ragcore.claims.types import ClaimRelation, SourceClaim


def _claim(claim_id: str, source_id: str, chunk_id: str, concept_id: str) -> SourceClaim:
    return SourceClaim(
        claim_id=claim_id, concept_ids=(concept_id,), source_id=source_id, chunk_id=chunk_id,
        text=f"text for {claim_id}", author_confidence="firm", model="fake", prompt_version="v1",
        created_at="now", vector=(1.0, 0.0),
    )


def test_remove_chunk_ids_deletes_the_claim_and_its_concept_associations() -> None:
    store = SqliteClaimStore(":memory:")
    c1 = _claim("c1", "srcA", "srcA:0", "concept:x")
    store.add_chapter_claims(source_id="srcA", context_group_id="ch1", fingerprint="fp1", claims=[c1])

    assert store.claim("c1") is not None
    assert store.concept_ids_with_claims(min_count=1) == ["concept:x"]

    store.remove_chunk_ids(["srcA:0"])

    assert store.claim("c1") is None
    assert store.concept_ids_with_claims(min_count=1) == []


def test_remove_chunk_ids_deletes_relations_touching_either_endpoint() -> None:
    store = SqliteClaimStore(":memory:")
    c1 = _claim("c1", "srcA", "srcA:0", "concept:x")
    c2 = _claim("c2", "srcB", "srcB:0", "concept:x")
    store.add_chapter_claims(source_id="srcA", context_group_id="ch1", fingerprint="fp1", claims=[c1])
    store.add_chapter_claims(source_id="srcB", context_group_id="ch1", fingerprint="fp2", claims=[c2])

    rid, a, b = _relation_id("c1", "c2", "conflicts")
    rel = ClaimRelation(
        relation_id=rid, source_claim_id=a, target_claim_id=b, relation_type="conflicts",
        rationale="r", model="fake", prompt_version="v1", created_at="now",
    )
    store.put_relations([rel])
    assert len(store.relations_for_claim("c2")) == 1

    # Removing c1's chunk (srcA is untouched otherwise) must delete the
    # relation entirely, even though c2 and its own source survive.
    store.remove_chunk_ids(["srcA:0"])

    assert store.claim("c1") is None
    assert store.claim("c2") is not None  # c2's own source is untouched
    assert store.relations_for_claim("c2") == []


def test_remove_source_prunes_all_its_claims_and_checkpoints() -> None:
    store = SqliteClaimStore(":memory:")
    c1 = _claim("c1", "srcA", "srcA:0", "concept:x")
    c2 = _claim("c2", "srcA", "srcA:1", "concept:x")
    store.add_chapter_claims(source_id="srcA", context_group_id="ch1", fingerprint="fp1", claims=[c1, c2])
    assert store.checkpoint("srcA", "ch1") == "fp1"

    store.remove_source("srcA")

    assert store.claim("c1") is None
    assert store.claim("c2") is None
    assert store.checkpoint("srcA", "ch1") is None


def test_removal_below_two_claims_garbage_collects_the_link_checkpoint() -> None:
    store = SqliteClaimStore(":memory:")
    c1 = _claim("c1", "srcA", "srcA:0", "concept:x")
    c2 = _claim("c2", "srcB", "srcB:0", "concept:x")
    store.add_chapter_claims(source_id="srcA", context_group_id="ch1", fingerprint="fp1", claims=[c1])
    store.add_chapter_claims(source_id="srcB", context_group_id="ch1", fingerprint="fp2", claims=[c2])
    store.mark_concept_linked(concept_id="concept:x", fingerprint="linked-fp")
    assert store.link_checkpoint("concept:x") == "linked-fp"

    store.remove_source("srcA")  # concept:x now has only 1 claim left

    assert store.link_checkpoint("concept:x") is None


def test_remove_chunk_ids_on_an_unknown_chunk_is_a_safe_no_op() -> None:
    store = SqliteClaimStore(":memory:")
    c1 = _claim("c1", "srcA", "srcA:0", "concept:x")
    store.add_chapter_claims(source_id="srcA", context_group_id="ch1", fingerprint="fp1", claims=[c1])

    store.remove_chunk_ids(["does-not-exist:0"])

    assert store.claim("c1") is not None
