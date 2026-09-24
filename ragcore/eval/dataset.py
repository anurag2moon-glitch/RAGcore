"""JSONL storage for GoldQuestion sets (one question per line — easy to
diff/append by hand), plus an authoring helper that resolves a literal
marker substring in real ingested chunk text into a concrete GoldPassage.

The marker-resolution technique mirrors
tests/integration/synth/library.py's find_chunks_with_marker, offered here
as a first-class, general-purpose authoring aid rather than a test-only
helper — reimplemented, not imported, since package code must never depend
on tests/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from ragcore.eval.types import GoldPassage, GoldQuestion

if TYPE_CHECKING:
    from ragcore.corpus import Corpus


def load_gold_set(path: str | Path) -> list[GoldQuestion]:
    questions: list[GoldQuestion] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(GoldQuestion.from_dict(json.loads(line)))
    return questions


def save_gold_set(path: str | Path, questions: Sequence[GoldQuestion]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q.to_dict()) + "\n")


def find_gold_passage(corpus: Corpus, source_title: str, marker: str) -> GoldPassage:
    """Authoring helper, run once interactively while building a gold set
    by hand — not needed at eval-run time. Scans the corpus for the first
    chunk containing the literal substring `marker`, in a source whose
    title equals `source_title`, and returns a GoldPassage pinned to that
    exact chunk.

    Raises ValueError if no matching chunk is found, since a silent empty
    GoldPassage would be a much harder authoring bug to notice than an
    immediate failure while writing the gold set.
    """
    for source in corpus.sources():
        if source.title != source_title:
            continue
        for chunk in corpus.store.get_by_source(source.source_id):
            if marker in chunk.text:
                return GoldPassage(
                    source_id=source.source_id,
                    source_path=source.path,
                    chapter_index=chunk.meta.chapter_index,
                    chunk_ids=(chunk.chunk_id,),
                )
    raise ValueError(f"no chunk containing {marker!r} found in a source titled {source_title!r}")
