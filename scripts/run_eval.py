"""Ad-hoc CLI for running the ragcore.eval harness against an existing
corpus and a gold-question JSONL file. Not part of the library's public
API — a convenience script, matching scripts/query.py's shape.

Usage:
    VOYAGE_API_KEY=... python scripts/run_eval.py <corpus-path> <questions.jsonl> [k]
    VOYAGE_API_KEY=... ANTHROPIC_API_KEY=... python scripts/run_eval.py <corpus-path> <questions.jsonl> [k] --verify
    VOYAGE_API_KEY=... python scripts/run_eval.py <corpus-path> <questions.jsonl> [k] --baseline old_report.json
    python scripts/run_eval.py --demo        # no API keys needed: builds a tiny
                                              # synthetic corpus and runs the
                                              # starter question set against it

--verify additionally scores groundedness — it requires ANTHROPIC_API_KEY
(it generates an answer and runs ClaudeVerifier against it, per question).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ragcore.corpus import Corpus
from ragcore.eval.dataset import load_gold_set
from ragcore.eval.report import compare, format_report, run_eval
from ragcore.eval.types import EvalReport

STARTER_QUESTIONS = Path(__file__).parent.parent / "ragcore" / "eval" / "examples" / "starter.jsonl"
DEMO_BOOK_PATH = Path("samples/eval_demo_book.md")

DEMO_BOOK_TEXT = """\
# Chapter 1: The Rise

The young captain's ambition consumed every decision. He believed that
unchecked ambition, pursued without restraint, would carry him to greatness
no matter the cost to those around him.

# Chapter 2: The Fall

By the second act, the captain's own unchecked ambition had isolated him
from every ally who once trusted him. His downfall came not from an enemy
but from the same drive that once made him great.

# Chapter 3: A Different Path

A rival commander chose duty over desire at every turn, and while her career
advanced more slowly, she kept the loyalty of everyone she led — a quiet
argument that restraint outlasts unchecked ambition.
"""


def _run_demo() -> None:
    from tests.fakes import FakeEmbedder, InMemoryStore

    DEMO_BOOK_PATH.parent.mkdir(exist_ok=True)
    DEMO_BOOK_PATH.write_text(DEMO_BOOK_TEXT)

    corpus = Corpus(InMemoryStore(dim=16), FakeEmbedder(dim=16))
    corpus.ingest(DEMO_BOOK_PATH)

    questions = load_gold_set(STARTER_QUESTIONS)
    report = run_eval(corpus, questions, k=5)
    print(f"(demo mode: FakeEmbedder is hash-based, scores aren't semantically meaningful)\n")
    print(format_report(report))


def main() -> None:
    args = list(sys.argv[1:])

    if "--demo" in args:
        _run_demo()
        return

    verify = "--verify" in args
    if verify:
        args.remove("--verify")

    baseline_path = None
    if "--baseline" in args:
        i = args.index("--baseline")
        baseline_path = Path(args[i + 1])
        del args[i : i + 2]

    if len(args) < 2:
        print(f"usage: {sys.argv[0]} <corpus-path> <questions.jsonl> [k] [--verify] [--baseline old_report.json]")
        print(f"       {sys.argv[0]} --demo")
        raise SystemExit(1)

    db_path, questions_path = args[0], args[1]
    k = int(args[2]) if len(args) > 2 else 10

    from ragcore.embedding.voyage import VoyageContextEmbedder
    from ragcore.stores.lancedb import LanceDBStore

    embedder = VoyageContextEmbedder(output_dimension=1024, min_request_interval=21.0)
    store = LanceDBStore(db_path, dim=embedder.dim)

    generator = None
    verifier = None
    if verify:
        from ragcore.generation.claude import ClaudeGenerator
        from ragcore.verification.claude import ClaudeVerifier

        generator = ClaudeGenerator()
        verifier = ClaudeVerifier()

    corpus = Corpus(store, embedder, generator=generator)
    questions = load_gold_set(questions_path)

    report = run_eval(corpus, questions, k=k, generator=generator, verifier=verifier)
    print(format_report(report))

    if baseline_path and baseline_path.exists():
        old_report = EvalReport.from_dict(json.loads(baseline_path.read_text()))
        print("\n--- vs baseline ---")
        print(compare(old_report, report))


if __name__ == "__main__":
    main()
