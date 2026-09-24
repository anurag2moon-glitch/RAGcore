"""ragcore.eval.groundedness: reuses ragcore.verification directly — this
test asserts on the composition (retrieve -> generate -> verify -> average),
not on real claim-checking logic, using a deterministic stub verifier."""

from __future__ import annotations

from pathlib import Path

from ragcore.corpus import Corpus
from ragcore.eval.groundedness import score_groundedness
from ragcore.eval.types import GoldQuestion
from ragcore.generation.base import Answer
from ragcore.verification.types import ClaimCheck, Verification
from tests.fakes import FakeEmbedder, FakeGenerator, InMemoryStore


class _StubVerifier:
    model = "stub"

    def __init__(self, ratio: float, unsupported_claims: tuple[str, ...] = ()) -> None:
        self._ratio = ratio
        self._unsupported = unsupported_claims
        self.calls: list[tuple[str, Answer]] = []

    def verify(self, query: str, answer: Answer) -> Verification:
        self.calls.append((query, answer))
        checks = [ClaimCheck(claim=c, status="unsupported", support=()) for c in self._unsupported]
        n_supported = max(0, round(self._ratio * 10)) if not checks else 0
        checks += [ClaimCheck(claim=f"s{i}", status="supported", support=("h:0",)) for i in range(n_supported)]
        return Verification(checks=tuple(checks) or (ClaimCheck(claim="x", status="supported", support=()),), model=self.model)


def test_score_groundedness_averages_supported_ratio_across_questions(md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    questions = [
        GoldQuestion(question_id="q1", query="a", gold_passages=()),
        GoldQuestion(question_id="q2", query="b", gold_passages=()),
    ]
    verifier = _StubVerifier(ratio=1.0)

    metrics, per_question = score_groundedness(corpus, questions, FakeGenerator(), verifier, k=3)

    assert metrics.n_questions_scored == 2
    assert len(verifier.calls) == 2
    assert "q1" in per_question and "q2" in per_question


def test_unsupported_claims_are_counted_and_surfaced(md_book: Path) -> None:
    corpus = Corpus(InMemoryStore(dim=8), FakeEmbedder(dim=8))
    corpus.ingest(md_book)
    questions = [GoldQuestion(question_id="q1", query="a", gold_passages=())]
    verifier = _StubVerifier(ratio=0.0, unsupported_claims=("bad claim 1", "bad claim 2"))

    metrics, per_question = score_groundedness(corpus, questions, FakeGenerator(), verifier, k=3)

    assert metrics.n_unsupported_claims == 2
    _ratio, unsupported = per_question["q1"]
    assert unsupported == ("bad claim 1", "bad claim 2")
