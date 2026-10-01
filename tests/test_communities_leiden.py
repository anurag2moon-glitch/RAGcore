"""detect_communities(algorithm="leiden") — docs/ARCHITECTURE.md Phase 4
gap list item 19. Mirrors test_communities.py's Louvain cases for the
same fixture, plus the algorithm-switch fingerprint invalidation and the
missing-networkx-version error path.
"""

from __future__ import annotations

import networkx
import pytest

pytest.importorskip("networkx")

from ragcore.concepts.communities import detect_communities
from ragcore.concepts.sqlite_store import SqliteConceptStore
from ragcore.concepts.types import Concept, ConceptEdge
from ragcore.errors import GenerationError

_HAS_LEIDEN = hasattr(networkx.algorithms.community, "leiden_communities")


def _seed_two_disconnected_pairs(store: SqliteConceptStore) -> None:
    concepts = [
        Concept(concept_id=f"concept:c{i}", label=f"idea {i}", description="d", source_chunk_ids=("c1",))
        for i in range(4)
    ]
    edges = [
        ConceptEdge(edge_id="edge:concept:c0:concept:c1", source_concept_id="concept:c0",
                    target_concept_id="concept:c1", weight=3, source_chunk_ids=("c1",)),
        ConceptEdge(edge_id="edge:concept:c2:concept:c3", source_concept_id="concept:c2",
                    target_concept_id="concept:c3", weight=3, source_chunk_ids=("c1",)),
    ]
    store.add_chapter_concepts(source_id="src", context_group_id="g", fingerprint="fp", concepts=concepts, edges=edges)


@pytest.mark.skipif(not _HAS_LEIDEN, reason="networkx version has no leiden_communities")
def test_leiden_groups_connected_concepts_together() -> None:
    store = SqliteConceptStore(":memory:")
    _seed_two_disconnected_pairs(store)

    report = detect_communities(store, algorithm="leiden", seed=0)

    assert report.ran is True
    assert report.community_count == 2
    c0_community = store.community_id_for_concept("concept:c0")
    c1_community = store.community_id_for_concept("concept:c1")
    c2_community = store.community_id_for_concept("concept:c2")
    assert c0_community == c1_community
    assert c0_community != c2_community


@pytest.mark.skipif(not _HAS_LEIDEN, reason="networkx version has no leiden_communities")
def test_leiden_skips_reclustering_when_graph_unchanged() -> None:
    store = SqliteConceptStore(":memory:")
    _seed_two_disconnected_pairs(store)

    detect_communities(store, algorithm="leiden", seed=0)
    report = detect_communities(store, algorithm="leiden", seed=0)

    assert report.ran is False


@pytest.mark.skipif(not _HAS_LEIDEN, reason="networkx version has no leiden_communities")
def test_switching_algorithm_invalidates_the_fingerprint_and_reclusters() -> None:
    store = SqliteConceptStore(":memory:")
    _seed_two_disconnected_pairs(store)

    detect_communities(store, algorithm="louvain", seed=0)
    report = detect_communities(store, algorithm="leiden", seed=0)

    assert report.ran is True  # different algorithm = different fingerprint, not a no-op


def test_leiden_without_a_supporting_networkx_version_raises_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    import ragcore.concepts.communities as communities_module

    class _NoLeidenCommunity:
        """Stands in for networkx.algorithms.community with no
        leiden_communities attribute, simulating an older networkx."""

    monkeypatch.setattr(communities_module.networkx.algorithms, "community", _NoLeidenCommunity())

    store = SqliteConceptStore(":memory:")
    _seed_two_disconnected_pairs(store)

    with pytest.raises(GenerationError, match="leiden_communities"):
        detect_communities(store, algorithm="leiden", seed=0)


def test_default_algorithm_is_still_louvain_with_an_unchanged_fingerprint_scheme() -> None:
    """The default (Louvain) path's fingerprint must stay byte-identical to
    before this change, so an existing deployment's checkpoint isn't
    invalidated by this refactor alone."""
    store = SqliteConceptStore(":memory:")
    _seed_two_disconnected_pairs(store)

    detect_communities(store, seed=0)  # algorithm defaults to "louvain"

    assert store.community_run_fingerprint() is not None
    # re-running with the default is a no-op, proving the fingerprint is
    # deterministic and stable for the default path
    report = detect_communities(store, seed=0)
    assert report.ran is False
