"""EntityExtractor implementation backed by Claude, using structured
outputs (Pydantic `output_format`, via `client.messages.parse()`) so
extraction returns validated objects instead of freeform text to parse.

Called once per ChunkGroup (chapter) by ragcore.graph.pipeline.build_graph
— not per chunk — specifically so pronoun/reference resolution ("he", "his
friend") has the whole chapter as context. Every input chunk is labeled
with a short numbered index (`[1]`, `[2]`, ...) in the prompt, matching the
citation pattern ragcore.generation.claude.ClaudeGenerator._format_context
already uses, and the model cites those numbers rather than real chunk_id
strings. Numbers are remapped back to real chunk_ids in `extract()` after
parsing — a real chunk_id (`{sha256}:{index}`) tokenizes to ~38 tokens per
citation versus ~1 for a bare number, and a chapter's worth of entities and
relations cites many of them, so this is a real, measured cost difference,
not a style preference. Entity/Relation.source_chunk_ids are unaffected —
they still hold real chunk_id strings; only the wire format between this
class and the model changed.

`Entity.entity_type` is open vocabulary at the protocol level
(ragcore/graph/types.py), but SYSTEM_PROMPT now names a concrete,
book-tuned taxonomy (character/person/place/work/organization/event) —
docs/ARCHITECTURE.md Phase 4 gap list item 16 — rather than the three
generic examples it suggested before. "theme" and "claim" are
deliberately *not* in this list: ragcore.concepts and ragcore.claims
already extract those at their own granularity and schema (a concept is
a proposition like "consequences of unchecked ambition," never a named
entity; a claim is an atomic, attributed assertion) — see CLAUDE.md's
concept-layer and claims-layer sections for why they're separate passes,
not alternate entity_type values here. Folding them into this
extractor's output would duplicate that work under a different, less
suited schema, not unify anything.
"""

from __future__ import annotations

import dataclasses
import re
import time
from typing import TYPE_CHECKING, Any, Sequence

from pydantic import BaseModel

from ragcore.errors import GenerationError
from ragcore.graph.merge import merge_extra
from ragcore.graph.types import Entity, Relation
from ragcore.model_routing import DEFAULT_ROUTING

if TYPE_CHECKING:
    # Deferred: ragcore.graph.__init__ imports this module, reached while
    # ragcore.types is still mid-import (it imports ragcore.graph.types for
    # GraphContext) — see ragcore/graph/pipeline.py for the full story.
    # Chunk is only ever used here as a type hint.
    from ragcore.types import Chunk

try:
    import anthropic
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    anthropic = None
else:
    _IMPORT_ERROR = None

SYSTEM_PROMPT = (
    "You extract named entities and relations between them from book text, for "
    "a knowledge graph. You are given every chunk of one chapter, in order, each "
    "labeled with a number in brackets, like [1]. Read them as a whole for "
    "context — resolve pronouns and referring phrases (\"he\", \"his friend\", "
    "\"her mother\") to the actual named entity they refer to using the "
    "surrounding text; never extract a pronoun itself as a name.\n\n"
    "For every entity: give its fullest canonical name (e.g. \"Mr. Bingley\", not "
    "\"he\" or just \"Bingley\"), a type, every distinct alias/surface form "
    "actually used to refer to it in this text (e.g. \"Bingley\", \"his friend\" "
    "— text strings that appear, not bare pronouns), and the bracketed number(s) "
    "of the chunk(s) where it is mentioned (e.g. 1, 3 — not the text \"[1]\", "
    "just the number itself).\n\n"
    "entity_type is open vocabulary, but prefer one of these when it fits: "
    "\"character\" (a named character in the narrative — the common case for "
    "fiction), \"person\" (a real-world/historical person the text refers to, "
    "distinct from a narrative character — common in nonfiction), \"place\", "
    "\"work\" (a named book, painting, ship, or other created/referenced work "
    "— not the entity reading/making/owning it), \"organization\", \"event\" "
    "(a specific, named happening, not a recurring kind of event). Use a "
    "different type only when none of these genuinely fits. Do not extract "
    "abstract themes or propositions (e.g. \"ambition\", \"the cost of "
    "betrayal\") as entities at all — a separate process handles those; you "
    "only extract things with a name.\n\n"
    "For every relation between two entities: give a relation_type (open "
    "vocabulary: \"spouse_of\", \"parent_of\", \"friend_of\", \"resides_at\", or "
    "a narrative-specific type if that fits better), reference the two entities "
    "by the exact canonical names you gave them, and the bracketed number(s) of "
    "the chunk(s) where the relation is stated or clearly implied.\n\n"
    "Only extract what this text actually supports. Do not invent relations "
    "between entities that are never connected in the given text."
)


class _ExtractedEntity(BaseModel):
    name: str
    entity_type: str
    aliases: list[str] = []
    chunk_indices: list[int]


class _ExtractedRelation(BaseModel):
    source_name: str
    target_name: str
    relation_type: str
    chunk_indices: list[int]


class _ExtractionResult(BaseModel):
    entities: list[_ExtractedEntity]
    relations: list[_ExtractedRelation]
    found_more: bool = True
    """Only meaningful on a gleaning round's response (see
    ClaudeEntityExtractor.extract) — the initial pass ignores it. Defaults
    to True so a non-gleaning call (which never reads this field) and an
    old recorded response both parse fine."""


GLEANING_PROMPT = (
    "Re-read the chapter text above carefully. List ONLY additional "
    "entities or relations you missed on your first pass — do not repeat "
    "anything already listed above. If you're confident you found "
    "everything, set found_more to false and leave entities/relations "
    "empty."
)


def _summarize_for_gleaning(entities: Sequence[Entity], relations: Sequence[Relation]) -> str:
    """A plain-text recap of what's been extracted so far, for the
    assistant turn of a gleaning round's conversation — not re-parsed, so
    it only needs to be clear to the model reviewing its own prior work,
    not machine-readable."""
    entity_lines = [f"- {e.name} ({e.entity_type})" for e in entities] or ["(none)"]
    relation_lines = [
        f"- {r.source_entity_id} {r.relation_type} {r.target_entity_id}" for r in relations
    ] or ["(none)"]
    return (
        "Entities found so far:\n"
        + "\n".join(entity_lines)
        + "\n\nRelations found so far:\n"
        + "\n".join(relation_lines)
    )


def _normalize(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "unknown"


def _entity_id(entity_type: str, name: str) -> str:
    return f"entity:{entity_type}:{_normalize(name)}"


def _format_chunks(chunks: Sequence[Chunk]) -> tuple[str, dict[int, str]]:
    """Returns the numbered prompt text plus the index->chunk_id mapping
    needed to resolve the model's citations back to real chunk_ids."""
    index_to_chunk_id = {i: c.chunk_id for i, c in enumerate(chunks, start=1)}
    text = "\n\n".join(f"[{i}]\n{c.text}" for i, c in enumerate(chunks, start=1))
    return text, index_to_chunk_id


class ClaudeEntityExtractor:
    """EntityExtractor implementation using Claude (cheap-tier by default —
    see ragcore.model_routing; extraction, not final-answer synthesis).

    `gleaning_rounds` (default 0, opt-in) adds up to that many follow-up
    "did you miss anything?" passes per `extract()` call — the gleaning
    technique from Microsoft's GraphRAG paper, for recall on a dense
    chapter a single pass under-extracts. A round stops early the moment
    the model reports nothing more found; 0 keeps `extract()`'s original
    single-pass behavior and cost exactly unchanged.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_ROUTING.cheap,
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 8192,
        max_retries: int = 5,
        gleaning_rounds: int = 0,
    ) -> None:
        if client is None and anthropic is None:
            raise GenerationError(
                "Claude entity extraction requires the 'generation' extra: "
                "install ragcore[generation]"
            ) from _IMPORT_ERROR

        self.model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._gleaning_rounds = gleaning_rounds
        self._client = client or (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def extract(self, chunks: Sequence[Chunk]) -> tuple[list[Entity], list[Relation]]:
        if not chunks:
            return [], []

        formatted, index_to_chunk_id = _format_chunks(chunks)

        def _resolve(indices: list[int]) -> tuple[str, ...]:
            # Silently drop any index outside the range we gave the model —
            # a hallucinated citation, not a crash-worthy error.
            return tuple(index_to_chunk_id[i] for i in indices if i in index_to_chunk_id)

        entities_by_id: dict[str, Entity] = {}
        relations_by_id: dict[str, Relation] = {}
        name_to_id: dict[str, str] = {}

        def _absorb(parsed: _ExtractionResult) -> None:
            for e in parsed.entities:
                entity_id = _entity_id(e.entity_type, e.name)
                name_to_id[e.name] = entity_id
                new_entity = Entity(
                    entity_id=entity_id,
                    name=e.name,
                    entity_type=e.entity_type,
                    source_chunk_ids=_resolve(e.chunk_indices),
                    extra={"aliases": e.aliases} if e.aliases else {},
                )
                existing = entities_by_id.get(entity_id)
                entities_by_id[entity_id] = (
                    new_entity
                    if existing is None
                    else dataclasses.replace(
                        existing,
                        source_chunk_ids=tuple(
                            sorted(set(existing.source_chunk_ids) | set(new_entity.source_chunk_ids))
                        ),
                        extra=merge_extra(existing.extra, new_entity.extra),
                    )
                )

            for r in parsed.relations:
                source_id = name_to_id.get(r.source_name)
                target_id = name_to_id.get(r.target_name)
                if source_id is None or target_id is None:
                    continue  # relation references an entity not extracted this call — drop it
                relation_id = f"relation:{r.relation_type}:{source_id}:{target_id}"
                new_relation = Relation(
                    relation_id=relation_id,
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relation_type=r.relation_type,
                    source_chunk_ids=_resolve(r.chunk_indices),
                )
                existing = relations_by_id.get(relation_id)
                relations_by_id[relation_id] = (
                    new_relation
                    if existing is None
                    else dataclasses.replace(
                        existing,
                        source_chunk_ids=tuple(
                            sorted(set(existing.source_chunk_ids) | set(new_relation.source_chunk_ids))
                        ),
                    )
                )

        messages: list[dict[str, Any]] = [{"role": "user", "content": formatted}]
        parsed = self._call_with_retry(messages)
        _absorb(parsed)

        for _ in range(self._gleaning_rounds):
            summary = _summarize_for_gleaning(entities_by_id.values(), relations_by_id.values())
            messages.append({"role": "assistant", "content": summary})
            messages.append({"role": "user", "content": GLEANING_PROMPT})
            gleaned = self._call_with_retry(messages)
            if not gleaned.found_more:
                break
            _absorb(gleaned)

        return list(entities_by_id.values()), list(relations_by_id.values())

    def _call_with_retry(self, messages: list[dict[str, Any]]) -> _ExtractionResult:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.parse(
                    model=self.model,
                    max_tokens=self._max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    output_format=_ExtractionResult,
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
                    raise GenerationError(f"{self.model} extraction request rejected: {e}") from e

            if attempt < self._max_retries - 1:
                time.sleep(min(2**attempt, 30))

        raise GenerationError(
            f"{self.model} entity extraction failed after {self._max_retries} attempts: {last_err}"
        ) from last_err
