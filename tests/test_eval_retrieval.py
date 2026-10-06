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


def test_score_retrieval_reports_real_latency_percentiles(md_book: Path) -> None:
    corpus = _corpus(md_book)
    questions = [
        GoldQuestion(question_id=f"q{i}", query="anything", gold_passages=()) for i in range(5)
    ]

    metrics, raw = score_retrieval(corpus, questions, k=3)

    assert metrics.latency_p50_ms >= 0.0
    assert metrics.latency_p95_ms >= metrics.latency_p50_ms
    assert metrics.latency_p99_ms >= metrics.latency_p95_ms
    assert all(latency_ms >= 0.0 for (_, _, _, latency_ms) in raw)


def test_score_retrieval_has_no_token_count_without_a_usage_tracking_embedder(md_book: Path) -> None:
    """FakeEmbedder (used throughout this test file) has no .usage
    attribute — total_tokens_used must stay None, never a guessed
    number, matching ragcore.embedding.cost.UsageTracker's own
    philosophy."""
    corpus = _corpus(md_book)
    questions = [GoldQuestion(question_id="q1", query="anything", gold_passages=())]

    metrics, _raw = score_retrieval(corpus, questions, k=3)

    assert metrics.total_tokens_used is None


def test_score_retrieval_sums_real_token_usage_when_available(md_book: Path) -> None:
    class _UsageTrackingEmbedder(FakeEmbedder):
        def __init__(self, dim: int) -> None:
            super().__init__(dim=dim)
            from ragcore.embedding.cost import UsageTracker

            self.usage = UsageTracker()

        def embed_query(self, text: str):
            self.usage.record_success(len(text.split()))
            return super().embed_query(text)

    embedder = _UsageTrackingEmbedder(dim=8)
    corpus = Corpus(InMemoryStore(dim=8), embedder)
    corpus.ingest(md_book)
    questions = [
        GoldQuestion(question_id="q1", query="three word query", gold_passages=()),
        GoldQuestion(question_id="q2", query="two words", gold_passages=()),
    ]

    metrics, _raw = score_retrieval(corpus, questions, k=3)

    assert metrics.total_tokens_used == 5  # 3 + 2 tokens, this run's queries only
