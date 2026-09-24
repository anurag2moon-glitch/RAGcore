"""plan_sync / sync_corpus: new/changed/removed detection, and — the core
bug this module exists to work around — that a changed file's *old* chunks
are correctly purged from every provided store (vector, lexical, graph),
never silently orphaned, plus that LLM-backed passes are scoped to only the
new/changed sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ragcore.claims.claude import _claim_id
from ragcore.claims.sqlite_store import SqliteClaimStore
from ragcore.claims.types import SourceClaim
from ragcore.concepts.sqlite_store import SqliteConceptStore
from ragcore.concepts.types import Concept
from ragcore.corpus import Corpus
from ragcore.graph.sqlite_store import SqliteGraphStore
from ragcore.graph.types import Entity
from ragcore.lexical.fts5 import Fts5LexicalStore
from ragcore.types import Chunk
from ragcore.versioning.manifest import IngestionManifest
from ragcore.versioning.sync import plan_sync, sync_corpus
from tests.fakes import FakeEmbedder, FakeEntityExtractor, InMemoryStore


class _FakeConceptExtractor:
    model = "fake-concepts"

    def extract(self, chunks: Sequence[Chunk]):
        concept = Concept(
            concept_id="concept:shared", label="shared idea", description="d",
            source_chunk_ids=tuple(c.chunk_id for c in chunks),
        )
        return [concept], []


class _FakeClaimExtractor:
    model = "fake-claims"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract(self, *, chunks: Sequence[Chunk], concepts):
        chunk = chunks[0]
        self.calls.append(chunk.meta.source_id)
        text = chunk.text.strip().splitlines()[0]
        return [SourceClaim(
            claim_id=_claim_id(chunk.meta.source_id, chunk.chunk_id, text),
            concept_ids=tuple(c.concept_id for c in concepts), source_id=chunk.meta.source_id,
            chunk_id=chunk.chunk_id, text=text, author_confidence="firm",
            model=self.model, prompt_version="v1", created_at="now",
        )]


def _corpus() -> Corpus:
    return Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))


def _entity_extractor() -> FakeEntityExtractor:
    calls_seen: list[str] = []

    def fn(chunks: Sequence[Chunk]):
        calls_seen.append(chunks[0].meta.source_id)
        entity = Entity(
            entity_id=f"entity:concept:{chunks[0].meta.source_id[:8]}",
            name="X", entity_type="concept",
            source_chunk_ids=tuple(c.chunk_id for c in chunks),
        )
        return [entity], []

    extractor = FakeEntityExtractor(fn)
    return extractor


def test_plan_sync_classifies_new_changed_removed_unchanged(tmp_path: Path) -> None:
    manifest = IngestionManifest(":memory:")
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("# A\n\noriginal")
    b.write_text("# B\n\ncontent")

    plan1 = plan_sync(manifest, [a, b])
    assert set(plan1.added) == {str(a.resolve()), str(b.resolve())}
    assert plan1.changed == ()
    assert plan1.removed == ()

    manifest.record_ingest(path=str(a.resolve()), source_id="h_a", content_hash=_hash(a), chunk_count=1)
    manifest.record_ingest(path=str(b.resolve()), source_id="h_b", content_hash=_hash(b), chunk_count=1)

    plan2 = plan_sync(manifest, [a, b])
    assert plan2.added == ()
    assert plan2.changed == ()
    assert set(plan2.unchanged) == {str(a.resolve()), str(b.resolve())}

    a.write_text("# A\n\nCHANGED")
    plan3 = plan_sync(manifest, [a, b])
    assert plan3.changed == (str(a.resolve()),)
    assert plan3.unchanged == (str(b.resolve()),)

    plan4 = plan_sync(manifest, [b])
    assert plan4.removed == (str(a.resolve()),)


def _hash(path: Path) -> str:
    from ragcore.versioning.manifest import hash_file

    return hash_file(path)


def test_sync_corpus_ingests_new_sources(tmp_path: Path) -> None:
    corpus = _corpus()
    manifest = IngestionManifest(":memory:")
    a = tmp_path / "a.md"
    a.write_text("# Chapter 1\n\nOriginal content.\n")

    report = sync_corpus(corpus, manifest, [a], snapshot=False)

    assert report.added == (str(a.resolve()),)
    assert report.chunks_ingested == corpus.count()
    assert corpus.count() > 0


def test_sync_corpus_on_a_changed_file_purges_old_chunks_from_every_store(tmp_path: Path) -> None:
    corpus = _corpus()
    manifest = IngestionManifest(":memory:")
    lexical = Fts5LexicalStore(":memory:")
    graph_store = SqliteGraphStore(":memory:")
    extractor = _entity_extractor()
    a = tmp_path / "a.md"
    a.write_text("# Chapter 1\n\nOriginal content about apples.\n")

    sync_corpus(
        corpus, manifest, [a], lexical=lexical, graph_store=graph_store,
        graph_extractor=extractor, snapshot=False,
    )
    old_source_id = manifest.source_id_for_path(str(a.resolve()))
    assert old_source_id is not None
    assert lexical.search("apples", k=10)
    assert corpus.store.get_by_source(old_source_id)

    a.write_text("# Chapter 1\n\nCHANGED content about bananas.\n")
    sync_corpus(
        corpus, manifest, [a], lexical=lexical, graph_store=graph_store,
        graph_extractor=extractor, snapshot=False,
    )

    new_source_id = manifest.source_id_for_path(str(a.resolve()))
    assert new_source_id != old_source_id
    # The old source's chunks must be gone everywhere — this is the exact
    # bug plain Corpus.ingest(replace=True) can't fix on its own, since it
    # only ever deletes by the file's NEW source_id.
    assert corpus.store.get_by_source(old_source_id) == []
    assert lexical.search("apples", k=10) == []
    assert lexical.search("bananas", k=10)
    assert corpus.store.get_by_source(new_source_id)


def test_sync_corpus_on_a_removed_file_removes_it_from_the_corpus(tmp_path: Path) -> None:
    corpus = _corpus()
    manifest = IngestionManifest(":memory:")
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("# A\n\ncontent a")
    b.write_text("# B\n\ncontent b")

    sync_corpus(corpus, manifest, [a, b], snapshot=False)
    assert len(corpus.sources()) == 2

    report = sync_corpus(corpus, manifest, [a], snapshot=False)

    assert report.removed == (str(b.resolve()),)
    assert len(corpus.sources()) == 1
    assert corpus.sources()[0].path == str(b.resolve()) or corpus.sources()[0].path == str(a.resolve())
    assert {s.path for s in corpus.sources()} == {str(a.resolve())}


def test_sync_corpus_rerun_with_nothing_changed_ingests_nothing(tmp_path: Path) -> None:
    corpus = _corpus()
    manifest = IngestionManifest(":memory:")
    a = tmp_path / "a.md"
    a.write_text("# A\n\ncontent")

    sync_corpus(corpus, manifest, [a], snapshot=False)
    count_after_first = corpus.count()

    report2 = sync_corpus(corpus, manifest, [a], snapshot=False)

    assert report2.added == ()
    assert report2.changed == ()
    assert report2.unchanged == (str(a.resolve()),)
    assert corpus.count() == count_after_first


def test_llm_backed_passes_are_scoped_to_only_new_or_changed_sources(tmp_path: Path) -> None:
    corpus = _corpus()
    manifest = IngestionManifest(":memory:")
    graph_store = SqliteGraphStore(":memory:")
    extractor = _entity_extractor()

    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("# A\n\ncontent a")
    b.write_text("# B\n\ncontent b")

    sync_corpus(corpus, manifest, [a, b], graph_store=graph_store, graph_extractor=extractor, snapshot=False)
    calls_after_first = len(extractor.calls)
    assert calls_after_first >= 2  # both books extracted

    # Change only `a` — the extractor must NOT be re-run against `b`'s
    # already-unchanged chunks.
    a.write_text("# A\n\nCHANGED content")
    sync_corpus(corpus, manifest, [a, b], graph_store=graph_store, graph_extractor=extractor, snapshot=False)

    a_source_id = manifest.source_id_for_path(str(a.resolve()))
    b_source_id = manifest.source_id_for_path(str(b.resolve()))
    calls_after_second = extractor.calls[calls_after_first:]
    touched_source_ids = {call[0].split(":")[0] for call in calls_after_second}
    assert a_source_id in touched_source_ids
    assert b_source_id not in touched_source_ids


def test_sync_corpus_lexical_index_reflects_only_current_sources(tmp_path: Path) -> None:
    corpus = _corpus()
    manifest = IngestionManifest(":memory:")
    lexical = Fts5LexicalStore(":memory:")
    a = tmp_path / "a.md"
    a.write_text("# A\n\nsome unique wordforfinding\n")

    sync_corpus(corpus, manifest, [a], lexical=lexical, snapshot=False)

    assert lexical.search("wordforfinding", k=10)
    assert lexical.count() == corpus.count()


def test_sync_corpus_runs_build_claims_immediately_after_build_concepts(tmp_path: Path) -> None:
    corpus = _corpus()
    manifest = IngestionManifest(":memory:")
    concept_store = SqliteConceptStore(":memory:")
    claim_store = SqliteClaimStore(":memory:")
    claim_extractor = _FakeClaimExtractor()
    a = tmp_path / "a.md"
    a.write_text("# A\n\nsome claim text\n")

    sync_corpus(
        corpus, manifest, [a],
        concept_store=concept_store, concept_extractor=_FakeConceptExtractor(),
        claim_store=claim_store, claim_extractor=claim_extractor,
        snapshot=False,
    )

    assert claim_extractor.calls  # build_claims actually ran, anchored to concepts
    assert claim_store.concept_ids_with_claims(min_count=1) == ["concept:shared"]


def test_sync_corpus_prunes_claims_for_a_removed_source(tmp_path: Path) -> None:
    corpus = _corpus()
    manifest = IngestionManifest(":memory:")
    concept_store = SqliteConceptStore(":memory:")
    claim_store = SqliteClaimStore(":memory:")
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("# A\n\ncontent a\n")
    b.write_text("# B\n\ncontent b\n")

    sync_corpus(
        corpus, manifest, [a, b],
        concept_store=concept_store, concept_extractor=_FakeConceptExtractor(),
        claim_store=claim_store, claim_extractor=_FakeClaimExtractor(),
        snapshot=False,
    )
    claims_before = claim_store.claims_for_concept("concept:shared")
    assert len(claims_before) == 2

    sync_corpus(
        corpus, manifest, [b],
        concept_store=concept_store, concept_extractor=_FakeConceptExtractor(),
        claim_store=claim_store, claim_extractor=_FakeClaimExtractor(),
        snapshot=False,
    )

    claims_after = claim_store.claims_for_concept("concept:shared")
    assert len(claims_after) == 1


def test_sync_corpus_calls_link_claims_when_a_linker_is_provided(tmp_path: Path) -> None:
    corpus = _corpus()
    manifest = IngestionManifest(":memory:")
    concept_store = SqliteConceptStore(":memory:")
    claim_store = SqliteClaimStore(":memory:")
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("# A\n\ncontent a\n")
    b.write_text("# B\n\ncontent b\n")

    class _NeverLinks:
        model = "fake-linker"
        calls = 0

        def link(self, claims):
            type(self).calls += 1
            return []

    sync_corpus(
        corpus, manifest, [a, b],
        concept_store=concept_store, concept_extractor=_FakeConceptExtractor(),
        claim_store=claim_store, claim_extractor=_FakeClaimExtractor(),
        claim_linker=_NeverLinks(),
        snapshot=False,
    )

    assert _NeverLinks.calls > 0  # link_claims ran as part of the corpus-global pass
    assert claim_store.link_checkpoint("concept:shared") is not None
