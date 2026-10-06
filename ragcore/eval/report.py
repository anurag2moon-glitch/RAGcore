"""run_eval — the top-level entry point tying retrieval scoring and
optional groundedness scoring into one EvalReport, plus a text renderer and
a diff helper for tracking impact across pipeline changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from ragcore.eval.groundedness import score_groundedness
from ragcore.eval.metrics import reciprocal_rank
from ragcore.eval.retrieval import score_retrieval
from ragcore.eval.types import EvalReport, GoldQuestion, QuestionResult

if TYPE_CHECKING:
    from ragcore.corpus import Corpus
    from ragcore.generation.base import Generator
    from ragcore.verification.base import Verifier


def run_eval(
    corpus: Corpus,
    questions: Sequence[GoldQuestion],
    *,
    k: int = 10,
    generator: Generator | None = None,
    verifier: Verifier | None = None,
) -> EvalReport:
    retrieval_metrics, raw = score_retrieval(corpus, questions, k=k)

    groundedness_metrics = None
    groundedness_by_question: dict[str, tuple[float, tuple[str, ...]]] = {}
    if generator is not None and verifier is not None:
        groundedness_metrics, groundedness_by_question = score_groundedness(
            corpus, questions, generator, verifier, k=k
        )

    per_question: list[QuestionResult] = []
    for (question_id, retrieved_ids, relevant, latency_ms), q in zip(raw, questions, strict=True):
        supported_ratio, unsupported = groundedness_by_question.get(question_id, (None, ()))
        per_question.append(
            QuestionResult(
                question_id=question_id,
                query=q.query,
                retrieved_chunk_ids=tuple(retrieved_ids),
                matched_gold_count=len(set(retrieved_ids) & relevant),
                gold_count=len(relevant),
                reciprocal_rank=reciprocal_rank(retrieved_ids, relevant),
                latency_ms=latency_ms,
                groundedness=supported_ratio,
                unsupported_claims=unsupported,
            )
        )

    return EvalReport(
        k=k,
        n_questions=len(questions),
        retrieval=retrieval_metrics,
        groundedness=groundedness_metrics,
        per_question=tuple(per_question),
    )


def format_report(report: EvalReport) -> str:
    r = report.retrieval
    lines = [
        f"n={report.n_questions} k={report.k}",
        f"  recall@k={r.recall_at_k:.3f}  precision@k={r.precision_at_k:.3f}  "
        f"ndcg@k={r.ndcg_at_k:.3f}  mrr={r.mrr:.3f}  hit_rate@k={r.hit_rate_at_k:.3f}",
        f"  latency: p50={r.latency_p50_ms:.1f}ms  p95={r.latency_p95_ms:.1f}ms  "
        f"p99={r.latency_p99_ms:.1f}ms"
        + (f"  tokens={r.total_tokens_used}" if r.total_tokens_used is not None else ""),
    ]
    if report.groundedness is not None:
        g = report.groundedness
        lines.append(
            f"  groundedness: mean_supported_ratio={g.mean_supported_ratio:.3f}  "
            f"unsupported_claims={g.n_unsupported_claims}  scored={g.n_questions_scored}"
        )
    return "\n".join(lines)


def compare(old: EvalReport, new: EvalReport) -> str:
    def _delta(a: float, b: float) -> str:
        d = b - a
        return f"{'+' if d >= 0 else ''}{d:.3f}"

    o, n = old.retrieval, new.retrieval
    lines = [
        f"recall@k:    {o.recall_at_k:.3f} -> {n.recall_at_k:.3f}  ({_delta(o.recall_at_k, n.recall_at_k)})",
        f"precision@k: {o.precision_at_k:.3f} -> {n.precision_at_k:.3f}  ({_delta(o.precision_at_k, n.precision_at_k)})",
        f"ndcg@k:      {o.ndcg_at_k:.3f} -> {n.ndcg_at_k:.3f}  ({_delta(o.ndcg_at_k, n.ndcg_at_k)})",
        f"mrr:         {o.mrr:.3f} -> {n.mrr:.3f}  ({_delta(o.mrr, n.mrr)})",
        f"hit_rate@k:  {o.hit_rate_at_k:.3f} -> {n.hit_rate_at_k:.3f}  ({_delta(o.hit_rate_at_k, n.hit_rate_at_k)})",
    ]
    if old.groundedness is not None and new.groundedness is not None:
        og, ng = old.groundedness, new.groundedness
        lines.append(
            f"groundedness: {og.mean_supported_ratio:.3f} -> {ng.mean_supported_ratio:.3f}  "
            f"({_delta(og.mean_supported_ratio, ng.mean_supported_ratio)})"
        )
    return "\n".join(lines)
