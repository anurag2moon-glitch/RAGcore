"""ClaudeGoldSetGenerator — synthesizes GoldQuestions directly from a
corpus's own ingested chapters (docs/ARCHITECTURE.md Phase 6 gap list
item 24: "synthetic gold-question generation from a real book"). Before
this, authoring a gold set meant hand-writing each GoldQuestion and
resolving it via `ragcore.eval.dataset.find_gold_passage` — a real,
useful authoring aid, but one that still needed a human to come up with
every question.

Same numbered-citation convention every other Claude-backed component in
this codebase uses (`ragcore.graph.claude_extractor._format_chunks`): the
model is given one chapter's chunks labeled `[1]`, `[2]`, ... and answers
with the bracketed number of the one chunk that most directly answers
each question it writes, never a real chunk_id string. This is what lets
`generate()` resolve a question straight to a concrete, correctly-scoped
`GoldPassage` with no further authoring step — the generated gold set is
immediately usable with `ragcore.eval.run_eval()`.

Called once per chapter (`context_group_id`), mirroring
`ragcore.graph.pipeline.build_graph`'s and
`ragcore.concepts.pipeline.build_concepts`'s chapter-scoping exactly, for
the same reason: a question grounded in one chapter's own content, not
requiring cross-chapter context to answer, is both easier for the model
to write well and a fairer unit test of retrieval (a gold question
answerable only by combining several chapters would conflate "did
retrieval find the right chunk" with "can multi-hop retrieval even
answer this," which is a different, already-covered concern —
see `ragcore.retrieval.iterative`).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Sequence

from pydantic import BaseModel

from ragcore.errors import GenerationError
from ragcore.eval.types import GoldPassage, GoldQuestion
from ragcore.graph.claude_extractor import _format_chunks
from ragcore.graph.pipeline import _group_by_context_group
from ragcore.model_routing import DEFAULT_ROUTING

if TYPE_CHECKING:
    from ragcore.corpus import Corpus

try:
    import anthropic
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    anthropic = None
else:
    _IMPORT_ERROR = None

SYSTEM_PROMPT = (
    "You write evaluation questions for a retrieval system, from one "
    "chapter of a book. You are given every chunk of the chapter, in "
    "order, each labeled with a number in brackets, like [1].\n\n"
    "Write natural questions a real reader might actually ask, each "
    "answerable using ONLY the content of ONE specific chunk from this "
    "chapter — not requiring outside context, other chapters, or combining "
    "multiple chunks. Avoid a question so generic it could be answered by "
    "many different chunks (e.g. 'what happens in this chapter' is too "
    "broad); prefer a question specific enough that exactly one chunk is "
    "the clear right answer.\n\n"
    "For each question: give the question itself, the bracketed number of "
    "the ONE chunk that most directly answers it (e.g. 3 — not the text "
    "'[3]', just the number), and 1-3 short factual statements the correct "
    "answer should mention."
)


class _GeneratedQuestion(BaseModel):
    query: str
    answer_chunk_index: int
    expected_facts: list[str] = []


class _GoldSetGenerationResult(BaseModel):
    questions: list[_GeneratedQuestion]


class ClaudeGoldSetGenerator:
    """GoldQuestion generator using Claude (cheap-tier by default — see
    ragcore.model_routing; generation of eval questions, not final-answer
    synthesis)."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_ROUTING.cheap,
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 4096,
        max_retries: int = 5,
        questions_per_chapter: int = 2,
    ) -> None:
        if client is None and anthropic is None:
            raise GenerationError(
                "Claude gold-set generation requires the 'generation' extra: "
                "install ragcore[generation]"
            ) from _IMPORT_ERROR

        self.model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._questions_per_chapter = questions_per_chapter
        self._client = client or (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def generate(
        self, corpus: Corpus, *, source_ids: Sequence[str] | None = None
    ) -> list[GoldQuestion]:
        sources = corpus.sources()
        if source_ids is not None:
            wanted = set(source_ids)
            sources = [s for s in sources if s.source_id in wanted]

        questions: list[GoldQuestion] = []
        for source in sources:
            chunks = corpus.store.get_by_source(source.source_id)
            for chapter_chunks in _group_by_context_group(chunks):
                formatted, index_to_chunk_id = _format_chunks(chapter_chunks)
                chunks_by_id = {c.chunk_id: c for c in chapter_chunks}

                parsed = self._call_with_retry(formatted)
                for generated in parsed.questions[: self._questions_per_chapter]:
                    chunk_id = index_to_chunk_id.get(generated.answer_chunk_index)
                    if chunk_id is None:
                        continue  # hallucinated citation — dropped, not a crash
                    chunk = chunks_by_id[chunk_id]
                    questions.append(
                        GoldQuestion(
                            question_id=f"synthetic:{chunk_id}:{len(questions)}",
                            query=generated.query,
                            gold_passages=(
                                GoldPassage(
                                    source_id=source.source_id,
                                    source_path=source.path,
                                    chapter_index=chunk.meta.chapter_index,
                                    chunk_ids=(chunk_id,),
                                ),
                            ),
                            expected_facts=tuple(generated.expected_facts),
                            notes="synthetic — generated by ClaudeGoldSetGenerator",
                        )
                    )

        return questions

    def _call_with_retry(self, user_content: str) -> _GoldSetGenerationResult:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.parse(
                    model=self.model,
                    max_tokens=self._max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                    output_format=_GoldSetGenerationResult,
                )
                return response.parsed_output
            except anthropic.RateLimitError as e:
                last_err = e
            except anthropic.APIConnectionError as e:
                last_err = e
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    last_err = e
                else:
                    raise GenerationError(f"{self.model} gold-set generation request rejected: {e}") from e

            if attempt < self._max_retries - 1:
                time.sleep(min(2**attempt, 30))

        raise GenerationError(
            f"{self.model} gold-set generation failed after {self._max_retries} attempts: {last_err}"
        ) from last_err
