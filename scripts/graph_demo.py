"""Runs the real build_graph() pipeline against the real, already-ingested
corpus (samples/library.lancedb) using a hand-authored fixture in place of
ClaudeEntityExtractor and the real Embedder — see
tests/pride_prejudice_fixture.py for what's being substituted and why.

No API calls anywhere: reading an existing LanceDB table needs no embedder
call, and the fixture's StubEmbedder stands in for Voyage in
EmbeddingEntityResolver.

Usage:
    python scripts/graph_demo.py
"""

from __future__ import annotations

from pathlib import Path

from ragcore.corpus import Corpus
from ragcore.graph.pipeline import build_graph
from ragcore.graph.sqlite_store import SqliteGraphStore
from ragcore.stores.lancedb import LanceDBStore
from tests.fakes import FakeEntityExtractor
from tests.pride_prejudice_fixture import StubEmbedder, build_resolver, extractor_fn

DB_PATH = Path("samples/library.lancedb")
GRAPH_PATH = Path("samples/graph.sqlite3")


def main() -> None:
    GRAPH_PATH.unlink(missing_ok=True)  # fresh graph each run of this demo

    stub_embedder = StubEmbedder()
    store = LanceDBStore(DB_PATH, dim=1024)
    corpus = Corpus(store, stub_embedder)  # embedder never called except by the resolver, below

    extractor = FakeEntityExtractor(extractor_fn)
    graph = SqliteGraphStore(GRAPH_PATH)
    resolver = build_resolver()

    report = build_graph(corpus, graph, extractor, resolver, source_ids=None)

    print("=== GraphReport ===")
    print(f"chunks_processed:     {report.chunks_processed}")
    print(f"entities_extracted:   {report.entities_extracted}  (candidates seen, pre-merge)")
    print(f"relations_extracted:  {report.relations_extracted}")
    print(f"entities_merged:      {report.entities_merged}  (resolver-driven merges)")
    print(f"extractor calls made: {len(extractor.calls)} (one per chapter with fixture data)")

    print("\n=== Entity list (post-resolution) ===")
    all_entities: dict[str, object] = {}
    for entity_type in ("person", "place"):
        for e in graph.entities_by_type(entity_type):
            all_entities[e.entity_id] = e
    for e in sorted(all_entities.values(), key=lambda e: (e.entity_type, e.name)):
        aliases = e.extra.get("aliases") or []
        alias_str = f"  (aka {', '.join(aliases)})" if aliases else ""
        chunk_ids = graph.chunks_for_entities([e.entity_id]).get(e.entity_id, [])
        print(f"  [{e.entity_type:6}] {e.name}{alias_str}")
        print(f"           id={e.entity_id}  chunks={chunk_ids}")

    print("\n=== Sample of relations (up to 10) ===")
    all_relations = graph.neighbors(list(all_entities.keys()), hops=1)
    seen_ids = set()
    unique_relations = []
    for r in all_relations:
        if r.relation_id not in seen_ids:
            seen_ids.add(r.relation_id)
            unique_relations.append(r)
    for r in unique_relations[:10]:
        src_name = all_entities[r.source_entity_id].name if r.source_entity_id in all_entities else r.source_entity_id
        tgt_name = all_entities[r.target_entity_id].name if r.target_entity_id in all_entities else r.target_entity_id
        chapters = sorted({c.meta.chapter_index for c in corpus.get(list(r.source_chunk_ids))})
        print(f"  {src_name} --{r.relation_type}--> {tgt_name}   chunks={r.source_chunk_ids}  chapter_index={chapters}")
    print(f"  ... ({len(unique_relations)} relations total)")

    print("\n=== Traversal: who is connected to Mr. Bennet, and how? ===")
    mr_bennet_id = "entity:person:mr-bennet"
    neighbor_relations = graph.neighbors([mr_bennet_id], hops=1)
    for r in neighbor_relations:
        other_id = r.target_entity_id if r.source_entity_id == mr_bennet_id else r.source_entity_id
        other = all_entities.get(other_id)
        other_name = other.name if other else other_id
        direction = "->" if r.source_entity_id == mr_bennet_id else "<-"
        print(f"  Mr. Bennet {direction} ({r.relation_type}) {direction} {other_name}")

    print("\n=== Merge outcomes to sanity-check ===")
    print('  "Bingley" (Ch I) + "Mr. Bingley" (Ch II/III) -> merged?',
          "entity:person:bingley" in all_entities and "entity:person:mr-bingley" not in all_entities)
    print('  "Elizabeth" (Ch II) + "Elizabeth Bennet" (Ch III) -> merged?',
          "entity:person:elizabeth" in all_entities and "entity:person:elizabeth-bennet" not in all_entities)
    print('  "Jane" (Ch I) + "the eldest Miss Bennet" (Ch III) -> merged?',
          "entity:person:jane" in all_entities and "entity:person:the-eldest-miss-bennet" not in all_entities)
    print("  (expected: first two True, third False — see fixture module docstring for why)")


if __name__ == "__main__":
    main()
