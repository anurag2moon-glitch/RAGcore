"""Demonstrates the conversation layer's debate mode: two personas, each
with their own separate Corpus, address a shared transcript via
DebateSession.take_turn() — a plain caller loop drives who speaks when
(DebateSession itself has no turn-taking policy, by design). Each persona's
reply is generated using ONLY that persona's own corpus; the opponent's
prior turn is visible to the query rewriter as conversational context, but
is never retrievable material for the other persona's answer.

No API calls: FakeEmbedder/InMemoryStore/FakeGenerator stand in for the
real pipeline, and a small deterministic fake stands in for the
Claude-backed rewriter.

Usage:
    python scripts/debate_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ragcore.conversation.session import DebateSession
from ragcore.conversation.transcript import InMemoryTranscriptStore
from ragcore.conversation.types import Turn
from ragcore.corpus import Corpus
from ragcore.generation.base import Answer
from ragcore.types import Retrieved
from tests.fakes import FakeEmbedder, InMemoryStore

SAMPLES_DIR = Path("samples")

STOIC_BOOK = SAMPLES_DIR / "debate_demo_stoic.md"
STOIC_TEXT = """\
# On Freedom

True freedom lies only in mastering one's own judgments. Nothing external
can be truly good or bad — only our reaction to it is within our power.
"""

EXISTENTIALIST_BOOK = SAMPLES_DIR / "debate_demo_existentialist.md"
EXISTENTIALIST_TEXT = """\
# On Freedom

Freedom is not mastery of judgment but the terrifying, total absence of a
given nature — we are condemned to choose ourselves, moment by moment,
with no external judge to answer to.
"""


class _FakeRewriter:
    model = "fake-rewriter"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def rewrite(self, *, transcript: Sequence[Turn], speaking_as: str) -> str:
        last = transcript[-1]
        if len(transcript) <= 1:
            rewritten = last.text
        else:
            rewritten = f"respond to: {last.text!r} (from {last.speaker})"
        self.calls.append((last.text, rewritten))
        return rewritten


class _FakeGenerator:
    model = "fake-generator"

    def generate(self, query: str, context: Sequence[Retrieved]) -> Answer:
        if not context:
            text = "I have nothing on this in my own writings."
        else:
            text = context[0].chunk.text.strip().splitlines()[-1]
        return Answer(text=text, citations=list(context))


def _persona_corpus(path: Path, text: str) -> Corpus:
    path.write_text(text)
    corpus = Corpus(InMemoryStore(dim=16), FakeEmbedder(dim=16))
    corpus.ingest(path)
    corpus.generator = _FakeGenerator()
    return corpus


def main() -> None:
    SAMPLES_DIR.mkdir(exist_ok=True)
    stoic = _persona_corpus(STOIC_BOOK, STOIC_TEXT)
    existentialist = _persona_corpus(EXISTENTIALIST_BOOK, EXISTENTIALIST_TEXT)

    debate = DebateSession(
        {"stoic": stoic, "existentialist": existentialist},
        transcript_store=InMemoryTranscriptStore(),
        rewriter=_FakeRewriter(),
    )
    debate.seed_topic("What is freedom?")

    speaking_order = ["stoic", "existentialist", "stoic", "existentialist"]
    for speaker in speaking_order:
        debate.take_turn(speaker)

    print("=== Debate transcript ===")
    for turn in debate.transcript():
        print(f"[{turn.speaker}] {turn.text}")

    debate.close()


if __name__ == "__main__":
    main()
