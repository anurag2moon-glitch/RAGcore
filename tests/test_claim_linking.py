"""link_claims: embedding-shortlist candidate bounding, batch-merge
idempotency (no duplicate relations from overlapping batches), resumability,
and — via ragcore.claims.claude._relation_id — symmetric vs. directional
relation-id canonicalization.
"""

from __future__ import annotations

from ragcore.claims.claude import _relation_id
from ragcore.claims.linking import _candidate_batches, link_claims
from ragcore.claims.sqlite_store import SqliteClaimStore
from ragcore.claims.types import ClaimRelation, SourceClaim


def _claim(cid: str, concept_id: str, text: str, vector: tuple[float, ...], confidence="firm") -> SourceClaim:
    return SourceClaim(
        claim_id=cid, concept_ids=(concept_id,), source_id=f"src-{cid}", chunk_id=f"src-{cid}:0",
        text=text, author_confidence=confidence, model="fake", prompt_version="v1", created_at="now",
        vector=vector,
    )


def _seed_store(claims: list[SourceClaim]) -> SqliteClaimStore:
    store = SqliteClaimStore(":memory:")
    by_chapter: dict[str, list[SourceClaim]] = {}
    for c in claims:
        by_chapter.setdefault(c.source_id, []).append(c)
    for i, (source_id, group) in enumerate(by_chapter.items()):
        store.add_chapter_claims(source_id=source_id, context_group_id="ch1", fingerprint=f"fp{i}", claims=group)
    return store


class _RecordingLinker:
    model = "fake-linker"

    def __init__(self, relation_fn) -> None:
        self._relation_fn = relation_fn
        self.batches: list[list[str]] = []

    def link(self, claims):
        self.batches.append([c.claim_id for c in claims])
        return self._relation_fn(claims)


def test_relation_id_canonicalization_is_symmetric_for_agrees_and_conflicts() -> None:
    id_ab, a1, b1 = _relation_id("claim:a", "claim:b", "conflicts")
    id_ba, a2, b2 = _relation_id("claim:b", "claim:a", "conflicts")
    assert id_ab == id_ba
    assert (a1, b1) == (a2, b2)


def test_relation_id_is_directional_for_extends() -> None:
    id_ab, source_ab, target_ab = _relation_id("claim:a", "claim:b", "extends")
    id_ba, source_ba, target_ba = _relation_id("claim:b", "claim:a", "extends")
    assert id_ab != id_ba
    assert source_ab == "claim:a" and target_ab == "claim:b"
    assert source_ba == "claim:b" and target_ba == "claim:a"


def test_concepts_with_fewer_than_two_claims_are_never_linked() -> None:
    claims = [_claim("c1", "concept:x", "only claim", (1.0, 0.0))]
    store = _seed_store(claims)
    linker = _RecordingLinker(lambda batch: [])

    report = link_claims(store, linker)

    assert report.concepts_processed == 0
    assert linker.batches == []


def test_link_claims_calls_the_linker_and_persists_relations() -> None:
    claims = [
        _claim("c1", "concept:x", "Freedom requires struggle.", (1.0, 0.0), "firm"),
        _claim("c2", "concept:x", "Freedom is given, not earned.", (0.9, 0.1), "hedged"),
    ]
    store = _seed_store(claims)

    def relation_fn(batch):
        ids = sorted(c.claim_id for c in batch)
        rid, a, b = _relation_id(ids[0], ids[1], "conflicts")
        return [ClaimRelation(
            relation_id=rid, source_claim_id=a, target_claim_id=b, relation_type="conflicts",
            rationale="opposing views on how freedom is attained", model="fake", prompt_version="v1",
            created_at="now",
        )]

    linker = _RecordingLinker(relation_fn)
    report = link_claims(store, linker)

    assert report.concepts_processed == 1
    assert report.relations_found == 1
    rels = store.relations_for_claim("c1")
    assert len(rels) == 1
    assert rels[0].relation_type == "conflicts"


def test_rerun_with_no_changes_skips_via_checkpoint() -> None:
    claims = [
        _claim("c1", "concept:x", "claim one", (1.0, 0.0)),
        _claim("c2", "concept:x", "claim two", (0.9, 0.1)),
    ]
    store = _seed_store(claims)
    linker = _RecordingLinker(lambda batch: [])

    report1 = link_claims(store, linker)
    assert report1.concepts_processed == 1
    calls_after_first = len(linker.batches)

    report2 = link_claims(store, linker)
    assert report2.concepts_skipped == 1
    assert report2.concepts_processed == 0
    assert len(linker.batches) == calls_after_first  # no re-linking


def test_candidate_batches_puts_every_claim_with_its_own_nearest_neighbors() -> None:
    # Two tight clusters far apart in vector space, within one concept group.
    cluster_a = [_claim(f"a{i}", "concept:x", f"a{i}", (1.0, 0.0 + i * 0.001)) for i in range(3)]
    cluster_b = [_claim(f"b{i}", "concept:x", f"b{i}", (0.0, 1.0 + i * 0.001)) for i in range(3)]
    claims = cluster_a + cluster_b

    batches = _candidate_batches(claims, shortlist_k=2, batch_size=20)

    assert batches  # nonempty
    all_ids = {c.claim_id for batch in batches for c in batch}
    assert all_ids == {c.claim_id for c in claims}

    # Every claim in cluster_a must share at least one batch with another
    # cluster_a claim (its nearest neighbors, by construction).
    for c in cluster_a:
        assert any(
            c.claim_id in {x.claim_id for x in batch}
            and any(o.claim_id in {y.claim_id for y in cluster_a} and o.claim_id != c.claim_id for o in batch)
            for batch in batches
        )


def test_overlapping_batches_produce_no_duplicate_relations() -> None:
    """A relation found redundantly by two overlapping batches must collapse
    to one row via content-addressed relation_id + put_relations' merge."""
    claims = [
        _claim("c1", "concept:x", "one", (1.0, 0.0)),
        _claim("c2", "concept:x", "two", (0.99, 0.01)),
        _claim("c3", "concept:x", "three", (0.98, 0.02)),
    ]
    store = _seed_store(claims)

    def relation_fn(batch):
        # Always report the same conflicts relation between c1 and c2,
        # regardless of which batch it's called with (as long as both are
        # present) — simulates the same real relation surfacing from
        # multiple overlapping candidate batches.
        ids = {c.claim_id for c in batch}
        if {"c1", "c2"} <= ids:
            rid, a, b = _relation_id("c1", "c2", "agrees")
            return [ClaimRelation(
                relation_id=rid, source_claim_id=a, target_claim_id=b, relation_type="agrees",
                rationale="r", model="fake", prompt_version="v1", created_at="now",
            )]
        return []

    linker = _RecordingLinker(relation_fn)
    link_claims(store, linker, shortlist_k=2, batch_size=2)

    assert len(store.relations_for_claim("c1")) <= 1
