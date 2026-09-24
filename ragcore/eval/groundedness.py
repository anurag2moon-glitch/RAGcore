"""Groundedness scoring: reuses ragcore.verification directly rather than
reimplementing any claim-checking logic — Verification.supported_ratio is
already a shipped, tested metric; this module just runs it per question and
averages the result.

Opt-in by construction: ragcore.eval.report.run_eval only calls this when
both a Generator and a Verifier are supplied. Omitting them yields free
retrieval-only metrics — deliberately, in place of a second, lower-fidelity
"cheap proxy" groundedness metric, which would itself be a kind of stand-in
this project avoids shipping.
"""

from __future__ import annotations

from statistics import fmean
from typing import TYPE_CHECKING, Sequence

from ragcore.eval.types import GoldQuestion, GroundednessMetrics

if TYPE_CHECKING:
    from ragcore.corpus import Corpus
    from ragcore.generation.base import Generator
    from ragcore.verification.base import Verifier


def score_groundedness(
    corpus: Corpus,
    questions: Sequence[GoldQuestion],
    generator: Generator,
    verifier: Verifier,
    *,
    k: int = 10,
) -> tuple[GroundednessMetrics, dict[str, tuple[float, tuple[str, ...]]]]:
    """Returns aggregate GroundednessMetrics plus a per-question_id map of
    (supported_ratio, unsupported claim texts), for run_eval to attach onto
    each QuestionResult without a second retrieve()+generate() pass."""
    ratios: list[float] = []
    n_unsupported = 0
    per_question: dict[str, tuple[float, tuple[str, ...]]] = {}

    for q in questions:
        results = corpus.retrieve(q.query, k=k)
        answer = generator.generate(q.query, results)
        verification = verifier.verify(q.query, answer)
        ratios.append(verification.supported_ratio)
        unsupported = tuple(c.claim for c in verification.unsupported)
        n_unsupported += len(unsupported)
        per_question[q.question_id] = (verification.supported_ratio, unsupported)

    metrics = GroundednessMetrics(
        mean_supported_ratio=fmean(ratios) if ratios else 1.0,
        n_unsupported_claims=n_unsupported,
        n_questions_scored=len(ratios),
    )
    return metrics, per_question
