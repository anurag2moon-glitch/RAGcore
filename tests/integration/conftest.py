"""Shared fixtures for the integration suite: the one synthetic library
(built once) and the one shared corpus/graph built from it (built once per
pytest session), so every test module exercises the exact same production-
scale index rather than each paying its own ingestion cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ragcore.corpus import Corpus
from ragcore.graph.pipeline import GraphReport, build_graph
from ragcore.graph.sqlite_store import SqliteGraphStore
from ragcore.stores.lancedb import LanceDBStore
from tests.integration.doubles import DeterministicEmbedder, EntityMention, MockEntityExtractor
from tests.integration.scale import is_quick
from tests.integration.synth.library import BookSpec, Library, build_library, resolve_mentions_to_groups

EMBED_DIM = 256


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-mark every test collected under tests/integration/ so `pytest -m
    integration` (or excluding it) works without hand-marking each test
    function — `gutenberg` stays an explicit per-test opt-in (see
    tests/integration/test_gutenberg.py) since it also needs real network/cache
    state, not just "lives under this directory"."""
    for item in items:
        if "tests/integration/" in str(item.fspath).replace("\\", "/"):
            item.add_marker(pytest.mark.integration)

# A representative subset for RAGCORE_IT_SCALE=quick: one small book, one
# with cross-book entity mentions, one with many chapters — enough to keep
# every test module's assertions meaningful, at a fraction of the ingest
# cost of the full 8-book library.
QUICK_BOOK_KEYS = ("very_short", "unusual_naming", "clean_toc", "fragment_anchored")


@dataclass
class SharedLibrary:
    corpus: Corpus
    graph: SqliteGraphStore
    library: Library
    extractor: MockEntityExtractor
    mentions_by_group: dict[str, list[EntityMention]]
    book_keys: tuple[str, ...]
    graph_report: GraphReport
    db_dir: str

    def book(self, key: str) -> BookSpec:
        return self.library.book(key)

    def is_ingested(self, key: str) -> bool:
        return key in self.book_keys


@pytest.fixture(scope="session")
def synth_library() -> Library:
    return build_library()


@pytest.fixture(scope="session")
def shared_library(tmp_path_factory: pytest.TempPathFactory, synth_library: Library) -> SharedLibrary:
    book_keys = QUICK_BOOK_KEYS if is_quick() else tuple(b.key for b in synth_library.books)

    db_dir = tmp_path_factory.mktemp("shared_library")
    embedder = DeterministicEmbedder(dim=EMBED_DIM)
    store = LanceDBStore(str(db_dir / "library.lancedb"), dim=EMBED_DIM)
    corpus = Corpus(store, embedder)

    for key in book_keys:
        corpus.ingest(synth_library.book(key).path)

    graph = SqliteGraphStore(str(db_dir / "graph.sqlite3"))
    mentions_by_group = resolve_mentions_to_groups(corpus, synth_library)
    extractor = MockEntityExtractor(mentions_by_group)
    report = build_graph(corpus, graph, extractor)

    return SharedLibrary(
        corpus=corpus,
        graph=graph,
        library=synth_library,
        extractor=extractor,
        mentions_by_group=mentions_by_group,
        book_keys=book_keys,
        graph_report=report,
        db_dir=str(db_dir),
    )


@pytest.fixture
def fresh_corpus(tmp_path, synth_library: Library) -> Corpus:
    """A bare, empty Corpus in an isolated tmp_path — for tests that
    mutate the index (re-ingest, remove, concurrent/failure injection) and
    so must NOT touch the session-scoped `shared_library`, which every
    other test module in this suite assumes stays intact and fully built.
    """
    embedder = DeterministicEmbedder(dim=EMBED_DIM)
    store = LanceDBStore(str(tmp_path / "corpus.lancedb"), dim=EMBED_DIM)
    return Corpus(store, embedder)
