"""General-purpose evaluation harness: a storable gold-question format
(book/chapter/chunk-id supporting passages), automated retrieval
precision/recall/nDCG/MRR scoring against them, and an opt-in groundedness
score reusing ragcore.verification — re-runnable after any pipeline change
to measure impact.

Distinct from tests/integration/eval/, which stays exactly as it is: a
harness deeply coupled to ragcore's own synthetic-library fixtures, for
ragcore's own regression testing, deliberately not part of this package
(see its own module docstring). This package is for a *consumer's* corpus
and questions.

    from ragcore.eval import (
        GoldPassage, GoldQuestion, RetrievalMetrics, GroundednessMetrics, EvalReport,
        load_gold_set, save_gold_set, find_gold_passage,
        run_eval, format_report, compare,
    )
"""

from ragcore.eval.dataset import find_gold_passage, load_gold_set, save_gold_set
from ragcore.eval.report import compare, format_report, run_eval
from ragcore.eval.types import (
    EvalReport,
    GoldPassage,
    GoldQuestion,
    GroundednessMetrics,
    QuestionResult,
    RetrievalMetrics,
)

__all__ = [
    "GoldPassage",
    "GoldQuestion",
    "RetrievalMetrics",
    "GroundednessMetrics",
    "QuestionResult",
    "EvalReport",
    "load_gold_set",
    "save_gold_set",
    "find_gold_passage",
    "run_eval",
    "format_report",
    "compare",
]
