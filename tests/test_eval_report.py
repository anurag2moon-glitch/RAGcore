"""ragcore.eval.report: run_eval() composing retrieval + optional
groundedness into one EvalReport, and the compare()/format_report()
rendering helpers. Also covers EvalReport.to_dict()/from_dict() round-trip,
since that's what makes a report re-runnable/diffable as a saved baseline.
"""

from __future__ import annotations

from pathlib import Path

from ragcore.corpus import Corpus
from ragcore.eval.report import compare, format_report, run_eval
from ragcore.eval.types import EvalReport, GoldPassage, GoldQuestion
from ragcore.generation.base import Answer
from ragcore.verification.types import ClaimCheck, Verification
from tests.fakes import FakeEmbedder, FakeGenerator, InMemoryStore


def _corpus(md_book: Path) -> Corpus:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    return corpus


class _AllSupportedVerifier:
    model = "stub"

    def verify(self, query: str, answer: Answer) -> Verification:
        return Verification(checks=(ClaimCheck(claim="c", status="supported", support=("h:0",)),), model=self.model)


def test_run_eval_retrieval_only_when_no_generator_or_verifier_given(md_book: Path) -> None:
    corpus = _corpus(md_book)
    chunk = corpus.store.all_chunks()[0]
    questions = [
        GoldQuestion(question_id="q1", query="anything", gold_passages=(GoldPassage(chunk_ids=(chunk.chunk_id,)),))
    ]

    report = run_eval(corpus, questions, k=len(corpus.store.all_chunks()))

    assert report.groundedness is None
    assert report.per_question[0].groundedness is None
    assert report.n_questions == 1


def test_run_eval_includes_groundedness_when_generator_and_verifier_given(md_book: Path) -> None:
    corpus = _corpus(md_book)
    questions = [GoldQuestion(question_id="q1", query="anything", gold_passages=())]

    report = run_eval(corpus, questions, k=5, generator=FakeGenerator(), verifier=_AllSupportedVerifier())

    assert report.groundedness is not None
    assert report.groundedness.mean_supported_ratio == 1.0
    assert report.per_question[0].groundedness == 1.0


def test_eval_report_to_dict_from_dict_round_trips(md_book: Path) -> None:
    corpus = _corpus(md_book)
    questions = [GoldQuestion(question_id="q1", query="anything", gold_passages=())]

    report = run_eval(corpus, questions, k=3)
    restored = EvalReport.from_dict(report.to_dict())

    assert restored.to_dict() == report.to_dict()


def test_format_report_includes_retrieval_metrics(md_book: Path) -> None:
    corpus = _corpus(md_book)
    report = run_eval(corpus, [GoldQuestion(question_id="q1", query="a", gold_passages=())], k=3)

    text = format_report(report)

    assert "recall@k" in text
    assert "mrr" in text


def test_compare_reports_a_delta_between_two_reports(md_book: Path) -> None:
    corpus = _corpus(md_book)
    chunk = corpus.store.all_chunks()[0]
    gp = GoldPassage(chunk_ids=(chunk.chunk_id,))

    old = run_eval(corpus, [GoldQuestion(question_id="q1", query="a", gold_passages=())], k=3)
    new = run_eval(corpus, [GoldQuestion(question_id="q1", query="a", gold_passages=(gp,))], k=3)

    text = compare(old, new)

    assert "recall@k" in text
    assert "->" in text
