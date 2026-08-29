"""Exercises build_graph() end to end against the real, already-ingested
Pride and Prejudice corpus (samples/library.lancedb — read-only, no
embedding call needed to read it), using the hand-authored fixture in
tests/pride_prejudice_fixture.py in place of ClaudeEntityExtractor and the
real Embedder. Zero API calls: no ANTHROPIC_API_KEY, no VOYAGE_API_KEY.

Skipped if the demo corpus hasn't been built (samples/ is gitignored).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("lancedb")

from ragcore.corpus import Corpus
from ragcore.graph.pipeline import build_graph
from ragcore.graph.sqlite_store import SqliteGraphStore
from ragcore.stores.lancedb import LanceDBStore
from tests.fakes import FakeEntityExtractor
from tests.pride_prejudice_fixture import build_resolver, extractor_fn

DB_PATH = Path("samples/library.lancedb")

pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(), reason="samples/library.lancedb not built in this environment"
)


def _run_build_graph():
    from tests.pride_prejudice_fixture import StubEmbedder

    corpus = Corpus(LanceDBStore(DB_PATH, dim=1024), StubEmbedder())
    extractor = FakeEntityExtractor(extractor_fn)
    graph = SqliteGraphStore(":memory:")
    report = build_graph(corpus, graph, extractor, build_resolver())
    return corpus, graph, extractor, report


def test_extraction_runs_once_per_chapter_not_per_chunk():
    corpus, graph, extractor, report = _run_build_graph()

    # 7 chapters ingested; the fixture only hand-authors 3, the rest return ([], [])
    assert len(extractor.calls) == 7
    assert report.chunks_processed == sum(len(c) for c in extractor.calls)

    # each call got every chunk of exactly one chapter, in order
    for call_chunk_ids in extractor.calls:
        chapters = {
            c.meta.chapter_index for c in corpus.get(call_chunk_ids)
        }
        assert len(chapters) == 1


def test_lexically_similar_cross_chapter_aliases_merge():
    _, graph, _, report = _run_build_graph()

    bingley = graph.entities_by_type("person")
    ids = {e.entity_id for e in bingley}
    assert "entity:person:bingley" in ids  # Ch I's id survives as canonical
    assert "entity:person:mr-bingley" not in ids  # Ch II/III's candidate id was absorbed

    assert "entity:person:elizabeth" in ids
    assert "entity:person:elizabeth-bennet" not in ids

    assert report.entities_merged == 3  # Bingley, Bingley (again in Ch III), Elizabeth


def test_lexically_dissimilar_alias_does_not_merge():
    """Honest negative case: a purely text-embedding resolver has no way to
    know "the eldest Miss Bennet" and "Jane" are the same person without
    wider context — both survive as separate entities."""
    _, graph, _, _ = _run_build_graph()

    ids = {e.entity_id for e in graph.entities_by_type("person")}
    assert "entity:person:jane" in ids
    assert "entity:person:the-eldest-miss-bennet" in ids


def test_exact_content_addressed_match_skips_resolver_entirely():
    """Mr. Bennet and Mrs. Bennet use the identical canonical string across
    all three chapters — they should merge via exact entity_id match alone,
    with zero resolver involvement (verified indirectly: entities_merged
    only counts the 3 lexical-alias cases, not these)."""
    _, graph, _, report = _run_build_graph()

    ids = {e.entity_id for e in graph.entities_by_type("person")}
    assert "entity:person:mr-bennet" in ids
    assert "entity:person:mrs-bennet" in ids

    mr_bennet_chunks = graph.chunks_for_entities(["entity:person:mr-bennet"])["entity:person:mr-bennet"]
    assert len(mr_bennet_chunks) == 2  # accumulated from both Ch I and Ch II, not overwritten


def test_edges_retain_provenance_to_chapter_index_and_chunk_id():
    corpus, graph, _, _ = _run_build_graph()

    relations = graph.neighbors(["entity:person:mr-darcy"], hops=1)
    slighted = next(r for r in relations if r.relation_type == "slighted")

    assert set(slighted.source_chunk_ids) == {
        "462be7852d84412c6695851395144a97e9762d45bd3c41b9f356dc7ac047b8a9:35",
        "462be7852d84412c6695851395144a97e9762d45bd3c41b9f356dc7ac047b8a9:36",
    }
    chapters = {c.meta.chapter_index for c in corpus.get(list(slighted.source_chunk_ids))}
    assert chapters == {4}  # Chapter III


def test_traversal_who_is_connected_to_mr_bennet():
    _, graph, _, _ = _run_build_graph()

    relations = graph.neighbors(["entity:person:mr-bennet"], hops=1)
    connections = {(r.relation_type, r.target_entity_id) for r in relations}

    assert ("spouse_of", "entity:person:mrs-bennet") in connections
    assert ("acquainted_with", "entity:person:bingley") in connections
    assert ("parent_of", "entity:person:elizabeth") in connections
    assert ("parent_of", "entity:person:kitty") in connections
    # Darcy is two hops out (via Bingley), not directly connected to Mr. Bennet
    assert not any(target == "entity:person:mr-darcy" for _rel, target in connections)


def test_relations_extracted_total_matches_fixture():
    _, _, _, report = _run_build_graph()
    assert report.relations_extracted == 12  # 2 + 4 + 6 across the three chapters
