"""ConceptExtractor implementation backed by Claude, using structured
outputs (Pydantic `output_format`, via `client.messages.parse()`) — the
exact same call shape as ragcore.graph.claude_extractor.ClaudeEntityExtractor.

Called once per ChunkGroup (chapter) by ragcore.concepts.pipeline.build_concepts
— not per chunk — for the same reason entity extraction is chapter-scoped:
giving the model a whole chapter lets it reuse one consistent label for the
same idea across chunks, which is what lets concepts within one chapter
collapse for free (exact-label match) without needing the resolver at all;
the resolver (ragcore.concepts.resolver) earns its cost across chapters and
books, where wording drifts. Reuses the numbered-citation scheme and
`_normalize` from ragcore.graph.claude_extractor rather than reimplementing
either.

Concept co-occurrence edges are never asked of the model — they are derived
here, after parsing, from which concepts the model attached to the same
chunk. Asking an LLM to also enumerate pairwise relationships between
however many concepts it just extracted is a needless O(n^2) prompt burden
for information already implicit in its own chunk_indices answer.
"""

from __future__ import annotations

import time
from itertools import combinations
from typing import TYPE_CHECKING, Any, Sequence

from pydantic import BaseModel

from ragcore.concepts.types import Concept, ConceptEdge
from ragcore.errors import GenerationError
from ragcore.graph.claude_extractor import _format_chunks, _normalize

if TYPE_CHECKING:
    from ragcore.types import Chunk

try:
    import anthropic
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    anthropic = None
else:
    _IMPORT_ERROR = None

SYSTEM_PROMPT = (
    "You extract abstract concepts and themes from book text, for a concept "
    "graph used in retrieval. You are given every chunk of one chapter, in "
    "order, each labeled with a number in brackets, like [1].\n\n"
    "A concept is an abstract IDEA or PROPOSITION the text discusses — never "
    "a bare keyword or topic label. For example, given text about a "
    "character's downfall caused by pride, extract \"consequences of "
    "unchecked ambition\" or \"pride precedes downfall\" — never just "
    "\"ambition\" or \"pride\". Given text contrasting duty and desire, "
    "extract \"tension between duty and personal desire\" — never just "
    "\"duty\". A good concept could be the thesis of a paragraph written "
    "about this passage; a bad concept is a word you could find with "
    "Ctrl-F.\n\n"
    "For each chunk, extract 2 to 5 distinct concepts it discusses. If the "
    "exact same idea recurs in multiple chunks of this chapter, reuse the "
    "identical label text for it every time rather than rephrasing — this "
    "is what lets repeated concepts collapse into one node instead of "
    "duplicating. For every concept: give a short label (a few words, the "
    "proposition itself), a one-sentence description that disambiguates it "
    "from similarly-worded ideas, and the bracketed number(s) of every "
    "chunk that discusses it (e.g. 1, 3 — not the text \"[1]\", just the "
    "number itself).\n\n"
    "Only extract concepts this text actually discusses. Do not extract "
    "concepts from outside knowledge about the book or its themes in "
    "general — ground every concept in the given chunks."
)


class _ExtractedConcept(BaseModel):
    label: str
    description: str
    chunk_indices: list[int]


class _ConceptResult(BaseModel):
    concepts: list[_ExtractedConcept]
    found_more: bool = True
    """Only meaningful on a gleaning round's response (see
    ClaudeConceptExtractor.extract) — the initial pass ignores it, same
    convention as ragcore.graph.claude_extractor._ExtractionResult."""


GLEANING_PROMPT = (
    "Re-read the chapter text above carefully. List ONLY additional "
    "concepts you missed on your first pass — do not repeat anything "
    "already listed above (reuse that exact label if a chunk you missed "
    "discusses one of those same ideas, rather than extracting it again "
    "under a new label). If you're confident you found everything, set "
    "found_more to false and leave concepts empty."
)


def _summarize_concepts_for_gleaning(concepts: Sequence[Concept]) -> str:
    lines = [f"- {c.label}: {c.description}" for c in concepts] or ["(none)"]
    return "Concepts found so far:\n" + "\n".join(lines)


def _concept_id(label: str) -> str:
    return f"concept:{_normalize(label)}"


def _edge_id(concept_a: str, concept_b: str) -> tuple[str, str, str]:
    """Returns (edge_id, source, target) with source < target, so an
    undirected edge always gets one canonical id regardless of extraction
    order."""
    a, b = sorted((concept_a, concept_b))
    return f"edge:{a}:{b}", a, b


def _derive_edges(concepts: list[Concept]) -> list[ConceptEdge]:
    """Co-occurrence: two concepts are edge-connected once per chunk they
    both appear in, with weight = number of such chunks."""
    concepts_by_chunk: dict[str, set[str]] = {}
    for c in concepts:
        for chunk_id in c.source_chunk_ids:
            concepts_by_chunk.setdefault(chunk_id, set()).add(c.concept_id)

    chunks_by_edge: dict[str, tuple[str, str, set[str]]] = {}
    for chunk_id, concept_ids in concepts_by_chunk.items():
        for a, b in combinations(sorted(concept_ids), 2):
            edge_id, source, target = _edge_id(a, b)
            _, _, chunks = chunks_by_edge.setdefault(edge_id, (source, target, set()))
            chunks.add(chunk_id)

    return [
        ConceptEdge(
            edge_id=edge_id,
            source_concept_id=source,
            target_concept_id=target,
            weight=len(chunks),
            source_chunk_ids=tuple(sorted(chunks)),
        )
        for edge_id, (source, target, chunks) in chunks_by_edge.items()
    ]


class ClaudeConceptExtractor:
    """ConceptExtractor implementation using Claude (claude-opus-5 by default).

    `gleaning_rounds` (default 0, opt-in): see
    ragcore.graph.claude_extractor.ClaudeEntityExtractor's identical
    parameter — same gleaning technique, same early-stop-on-no-more-found
    behavior, applied here to concepts instead of entities/relations.
    """

    def __init__(
        self,
        *,
        model: str = "claude-opus-5",
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 8192,
        max_retries: int = 5,
        gleaning_rounds: int = 0,
    ) -> None:
        if client is None and anthropic is None:
            raise GenerationError(
                "Claude concept extraction requires the 'generation' extra: "
                "install ragcore[generation]"
            ) from _IMPORT_ERROR

        self.model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._gleaning_rounds = gleaning_rounds
        self._client = client or (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def extract(self, chunks: Sequence[Chunk]) -> tuple[list[Concept], list[ConceptEdge]]:
        if not chunks:
            return [], []

        formatted, index_to_chunk_id = _format_chunks(chunks)

        def _resolve(indices: list[int]) -> set[str]:
            # Silently drop any index outside the range we gave the model —
            # a hallucinated citation, not a crash-worthy error. Matches
            # ragcore.graph.claude_extractor's convention exactly.
            return {index_to_chunk_id[i] for i in indices if i in index_to_chunk_id}

        # Merge extracted entries sharing a normalized label into one
        # Concept — the model is instructed to reuse identical label text
        # for a recurring idea, but this also collapses near-duplicates
        # that only differ by normalization-insignificant characters, and
        # (for a gleaning round) a concept re-found under the same label.
        merged: dict[str, dict[str, Any]] = {}

        def _absorb(parsed: _ConceptResult) -> None:
            for c in parsed.concepts:
                concept_id = _concept_id(c.label)
                entry = merged.setdefault(
                    concept_id, {"label": c.label, "description": c.description, "chunk_ids": set()}
                )
                entry["chunk_ids"] |= _resolve(c.chunk_indices)

        messages: list[dict[str, Any]] = [{"role": "user", "content": formatted}]
        parsed = self._call_with_retry(messages)
        _absorb(parsed)

        for _ in range(self._gleaning_rounds):
            current_concepts = [
                Concept(
                    concept_id=cid,
                    label=e["label"],
                    description=e["description"],
                    source_chunk_ids=tuple(sorted(e["chunk_ids"])),
                )
                for cid, e in merged.items()
            ]
            messages.append(
                {"role": "assistant", "content": _summarize_concepts_for_gleaning(current_concepts)}
            )
            messages.append({"role": "user", "content": GLEANING_PROMPT})
            gleaned = self._call_with_retry(messages)
            if not gleaned.found_more:
                break
            _absorb(gleaned)

        concepts = [
            Concept(
                concept_id=concept_id,
                label=entry["label"],
                description=entry["description"],
                source_chunk_ids=tuple(sorted(entry["chunk_ids"])),
            )
            for concept_id, entry in merged.items()
            if entry["chunk_ids"]  # a concept every citation of which was hallucinated has nothing to link
        ]

        return concepts, _derive_edges(concepts)

    def _call_with_retry(self, messages: list[dict[str, Any]]) -> _ConceptResult:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.parse(
                    model=self.model,
                    max_tokens=self._max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    output_format=_ConceptResult,
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
                    raise GenerationError(f"{self.model} concept extraction request rejected: {e}") from e

            if attempt < self._max_retries - 1:
                time.sleep(min(2**attempt, 30))

        raise GenerationError(
            f"{self.model} concept extraction failed after {self._max_retries} attempts: {last_err}"
        ) from last_err
