"""ragcore.eval.retrieval: resolving GoldPassage entries into concrete
chunk_id sets at each supported granularity."""

from __future__ import annotations

from pathlib import Path

from ragcore.corpus import Corpus
from ragcore.eval.retrieval import resolve_relevant_chunk_ids, score_retrieval
from ragcore.eval.types import GoldPassage, GoldQuestion
from tests.fakes import FakeEmbedder, InMemoryStore


def _corpus(md_book: Path) -> Corpus:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    return corpus


def test_exact_chunk_ids_are_used_directly(md_book: Path) -> None:
    corpus = _corpus(md_book)
    chunk = corpus.store.all_chunks()[0]

    relevant = resolve_relevant_chunk_ids(corpus, [GoldPassage(chunk_ids=(chunk.chunk_id,))])

    assert relevant == {chunk.chunk_id}


def test_source_and_chapter_index_resolves_to_every_chunk_in_that_chapter(md_book: Path) -> None:
    corpus = _corpus(md_book)
    all_chunks = corpus.store.all_chunks()
    target_chapter = all_chunks[0].meta.chapter_index
    source_id = all_chunks[0].meta.source_id
    expected = {c.chunk_id for c in all_chunks if c.meta.chapter_index == target_chapter}

    relevant = resolve_relevant_chunk_ids(
        corpus, [GoldPassage(source_id=source_id, chapter_index=target_chapter)]
    )

    assert relevant == expected
    assert relevant  # test premise: at least one chunk in that chapter


def test_source_path_alone_resolves_to_every_chunk_in_that_source(md_book: Path) -> None:
    corpus = _corpus(md_book)
    source = corpus.sources()[0]

    relevant = resolve_relevant_chunk_ids(corpus, [GoldPassage(source_path=source.path)])

    expected = {c.chunk_id for c in corpus.store.get_by_source(source.source_id)}
    assert relevant == expected


def test_unresolvable_gold_passage_contributes_nothing_without_raising(md_book: Path) -> None:
    corpus = _corpus(md_book)

    relevant = resolve_relevant_chunk_ids(corpus, [GoldPassage(source_path="/nonexistent/path.md")])

    assert relevant == set()


def test_score_retrieval_aggregates_across_questions(md_book: Path) -> None:
    corpus = _corpus(md_book)
    all_chunks = corpus.store.all_chunks()
    target = all_chunks[0]

    questions = [
        GoldQuestion(
            question_id="q1", query="anything",
            gold_passages=(GoldPassage(chunk_ids=(target.chunk_id,)),),
        )
    ]

    metrics, raw = score_retrieval(corpus, questions, k=len(all_chunks))

    assert metrics.k == len(all_chunks)
    assert metrics.recall_at_k == 1.0  # k covers every chunk, so the gold chunk is always retrieved
    assert len(raw) == 1
    assert raw[0][0] == "q1"
