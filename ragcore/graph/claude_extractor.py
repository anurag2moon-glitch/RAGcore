"""EntityExtractor implementation backed by Claude, using structured
outputs (Pydantic `output_format`, via `client.messages.parse()`) so
extraction returns validated objects instead of freeform text to parse.

Called once per ChunkGroup (chapter) by ragcore.graph.pipeline.build_graph
— not per chunk — specifically so pronoun/reference resolution ("he", "his
friend") has the whole chapter as context. Every input chunk is labeled
with its real chunk_id in the prompt, and the model is required to cite
those exact chunk_id strings on every entity and relation it extracts —
the same grounded-citation pattern already used in
ragcore.generation.claude.ClaudeGenerator._format_context.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any, Sequence

from pydantic import BaseModel

from ragcore.errors import GenerationError
from ragcore.graph.types import Entity, Relation

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
    "labeled with its chunk_id. Read them as a whole for context — resolve "
    "pronouns and referring phrases (\"he\", \"his friend\", \"her mother\") to "
    "the actual named entity they refer to using the surrounding text; never "
    "extract a pronoun itself as a name.\n\n"
    "For every entity: give its fullest canonical name (e.g. \"Mr. Bingley\", not "
    "\"he\" or just \"Bingley\"), a type (open vocabulary: \"person\", \"place\", "
    "\"event\", or whatever fits), every distinct alias/surface form actually "
    "used to refer to it in this text (e.g. \"Bingley\", \"his friend\" — text "
    "strings that appear, not bare pronouns), and the chunk_id(s) (from the "
    "labels below) where it is mentioned.\n\n"
    "For every relation between two entities: give a relation_type (open "
    "vocabulary: \"spouse_of\", \"parent_of\", \"friend_of\", \"resides_at\", or "
    "a narrative-specific type if that fits better), reference the two entities "
    "by the exact canonical names you gave them, and the chunk_id(s) where the "
    "relation is stated or clearly implied.\n\n"
    "Only extract what this text actually supports. Do not invent relations "
    "between entities that are never connected in the given text."
)


class _ExtractedEntity(BaseModel):
    name: str
    entity_type: str
    aliases: list[str] = []
    chunk_ids: list[str]


class _ExtractedRelation(BaseModel):
    source_name: str
    target_name: str
    relation_type: str
    chunk_ids: list[str]


class _ExtractionResult(BaseModel):
    entities: list[_ExtractedEntity]
    relations: list[_ExtractedRelation]


def _normalize(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "unknown"


def _entity_id(entity_type: str, name: str) -> str:
    return f"entity:{entity_type}:{_normalize(name)}"


def _format_chunks(chunks: Sequence[Chunk]) -> str:
    return "\n\n".join(f"[chunk_id: {c.chunk_id}]\n{c.text}" for c in chunks)


class ClaudeEntityExtractor:
    """EntityExtractor implementation using Claude (claude-opus-5 by default)."""

    def __init__(
        self,
        *,
        model: str = "claude-opus-5",
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 8192,
        max_retries: int = 5,
    ) -> None:
        if client is None and anthropic is None:
            raise GenerationError(
                "Claude entity extraction requires the 'generation' extra: "
                "install ragcore[generation]"
            ) from _IMPORT_ERROR

        self.model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._client = client or (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def extract(self, chunks: Sequence[Chunk]) -> tuple[list[Entity], list[Relation]]:
        if not chunks:
            return [], []

        parsed = self._call_with_retry(_format_chunks(chunks))

        entities: list[Entity] = []
        name_to_id: dict[str, str] = {}
        for e in parsed.entities:
            entity_id = _entity_id(e.entity_type, e.name)
            name_to_id[e.name] = entity_id
            entities.append(
                Entity(
                    entity_id=entity_id,
                    name=e.name,
                    entity_type=e.entity_type,
                    source_chunk_ids=tuple(e.chunk_ids),
                    extra={"aliases": e.aliases} if e.aliases else {},
                )
            )

        relations: list[Relation] = []
        for r in parsed.relations:
            source_id = name_to_id.get(r.source_name)
            target_id = name_to_id.get(r.target_name)
            if source_id is None or target_id is None:
                continue  # relation references an entity not extracted this call — drop it
            relations.append(
                Relation(
                    relation_id=f"relation:{r.relation_type}:{source_id}:{target_id}",
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relation_type=r.relation_type,
                    source_chunk_ids=tuple(r.chunk_ids),
                )
            )

        return entities, relations

    def _call_with_retry(self, user_content: str) -> _ExtractionResult:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.parse(
                    model=self.model,
                    max_tokens=self._max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
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
