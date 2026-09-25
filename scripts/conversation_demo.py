"""Demonstrates the conversation layer's chat mode: PersonaChatSession
wraps one persona's existing Corpus with a session transcript and
context-aware query rewriting. Turn 3 ("what about when I feel guilty
about it?") only resolves to something retrieval can act on because the
rewriter resolves it against the transcript first — this prints both the
raw message and the rewritten query per turn to make that visible.

No API calls: FakeEmbedder/InMemoryStore/FakeGenerator stand in for the
real pipeline, and a small deterministic fake stands in for the Claude-backed
rewriter — the same role every other *_demo.py script's fakes play.

Usage:
    python scripts/conversation_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ragcore.conversation.session import PersonaChatSession
from ragcore.conversation.transcript import InMemoryTranscriptStore
from ragcore.conversation.types import Turn
from ragcore.corpus import Corpus
from ragcore.generation.base import Answer
from ragcore.types import Retrieved
from tests.fakes import FakeEmbedder, InMemoryStore

SAMPLES_DIR = Path("samples")
BOOK_PATH = SAMPLES_DIR / "conversation_demo_book.md"

BOOK_TEXT = """\
# On Virtue

Virtue is a kind of knowledge — one who truly knows the good cannot help
but do it. Wrongdoing is always a failure of understanding, not of will.

# On Guilt

Guilt is the soul's own recognition that it has acted against what it
knew to be good. It is not a punishment imposed from outside, but evidence
that the capacity for virtue is still alive and has not been silenced.
"""


class _FakeRewriter:
    """Deterministic stand-in for HistoryAwareQueryRewriter: pretends to
    resolve an implicit reference ("it", "that") to the most recent topic
    the transcript actually discussed, so the demo can show a concrete
    before/after without an API call. Records each (raw, rewritten) pair it
    produces so the demo can print exactly what it did, without recomputing
    it a second time."""

    model = "fake-rewriter"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def rewrite(self, *, transcript: Sequence[Turn], speaking_as: str) -> str:
        raw = transcript[-1].text
        if len(transcript) <= 1:
            rewritten = raw
        elif "it" in raw.lower() or "that" in raw.lower():
            topic = next(
                (t.text for t in reversed(transcript[:-1]) if t.speaker == "user"), ""
            )
            rewritten = f"{raw} (regarding: {topic})"
        else:
            rewritten = raw
        self.calls.append((raw, rewritten))
        return rewritten


class _FakeGenerator:
    """Deterministic Generator double that answers using whichever chunk
    text was actually retrieved, so the printed answer visibly changes
    based on what the (rewritten) query found."""

    model = "fake-generator"

    def generate(self, query: str, context: Sequence[Retrieved]) -> Answer:
        if not context:
            text = "I have nothing on that in my own writings."
        else:
            text = f"On this matter, I would say: {context[0].chunk.text.strip()}"
        return Answer(text=text, citations=list(context))


def main() -> None:
    SAMPLES_DIR.mkdir(exist_ok=True)
    BOOK_PATH.write_text(BOOK_TEXT)

    corpus = Corpus(InMemoryStore(dim=16), FakeEmbedder(dim=16))
    corpus.ingest(BOOK_PATH)
    corpus.generator = _FakeGenerator()

    rewriter = _FakeRewriter()
    session = PersonaChatSession(
        "socrates",
        corpus,
        transcript_store=InMemoryTranscriptStore(),
        rewriter=rewriter,
    )

    messages = [
        "What is virtue?",
        "Where does wrongdoing come from, then?",
        "What about when I feel guilty about it?",
    ]

    for message in messages:
        answer = session.send(message)
        raw, rewritten = rewriter.calls[-1]
        print(f"user:     {raw!r}")
        print(f"rewritten:{rewritten!r}")
        print(f"socrates: {answer.text!r}\n")

    session.close()


if __name__ == "__main__":
    main()
