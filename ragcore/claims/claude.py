"""ClaimExtractor and ClaimLinker implementations backed by Claude, using
structured outputs (Pydantic `output_format`, via `client.messages.parse()`)
— the same call shape every other Claude-backed component in this codebase
uses.

ClaudeClaimExtractor is called once per ChunkGroup (chapter) by
ragcore.claims.pipeline.build_claims, given both the chapter's chunks and
its already-extracted concepts (see ragcore.claims.base.ClaimExtractor for
why concepts are supplied rather than looked up after the fact).

ClaudeClaimLinker is called once per candidate batch by
ragcore.claims.linking.link_claims, never once per concept's full claim
set — see that module for the embedding-shortlist bounding strategy.
"""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Sequence

from pydantic import BaseModel

from ragcore.claims.types import ClaimConfidence, ClaimRelation, ClaimRelationType, SourceClaim
from ragcore.errors import GenerationError
from ragcore.graph.claude_extractor import _format_chunks
from ragcore.model_routing import DEFAULT_ROUTING

if TYPE_CHECKING:
    from ragcore.concepts.types import Concept
    from ragcore.types import Chunk

try:
    import anthropic
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    anthropic = None
else:
    _IMPORT_ERROR = None

EXTRACTOR_PROMPT_VERSION = "claims-v1"
LINKER_PROMPT_VERSION = "claim-links-v1"

EXTRACTOR_SYSTEM_PROMPT = (
    "You extract atomic factual or philosophical claims from book text, for "
    "a claim graph used to later compare how different passages address the "
    "same idea. You are given every chunk of one chapter, each labeled with "
    "a number in brackets like [1], and a numbered menu of concepts already "
    "identified in this chapter, each labeled like (1).\n\n"
    "A claim is one atomic, verifiable-in-principle assertion the text "
    "makes — a single proposition, not a whole paragraph's worth of "
    "argument. Quote it verbatim from the text (a short span, not a "
    "paraphrase). For each claim, give: the exact chunk number it comes "
    "from; every concept number(s) it is actually about (a claim with no "
    "relevant concept in the menu should not be extracted at all); and "
    "author_confidence: \"firm\" if the author states it directly and "
    "flatly in that passage, or \"hedged\" if the author's own wording in "
    "that exact passage signals uncertainty or tentativeness (\"perhaps\", "
    "\"I suspect\", \"it seems to me\", \"one might say\"). Judge "
    "author_confidence ONLY from this passage's own language — never from "
    "whether it agrees or disagrees with anything said elsewhere.\n\n"
    "Only extract claims this text actually makes. Do not extract claims "
    "from outside knowledge, and do not invent a concept link that isn't "
    "genuinely what the claim is about."
)

LINKER_SYSTEM_PROMPT = (
    "You are given a numbered list of claims, all about the same concept, "
    "each labeled with a number in brackets like [1] and its source and "
    "author_confidence. Find every pair that has a genuine relationship:\n\n"
    "- \"agrees\": the two claims assert compatible, mutually reinforcing "
    "positions.\n"
    "- \"conflicts\": the two claims assert incompatible positions on the "
    "same question — a reader could not hold both as stated.\n"
    "- \"extends\": one claim refines, develops, or adds nuance to the "
    "other without contradicting it (specify which is the source, i.e. the "
    "one doing the extending, and which is the target).\n\n"
    "Only report pairs with a real, direct relationship — most pairs of "
    "claims about the same concept have no relationship at all, and it is "
    "fine to report none. For every relation, give the two claim numbers, "
    "the relation type, and a one-sentence rationale grounded in what each "
    "claim actually says."
)


class _ExtractedClaim(BaseModel):
    chunk_index: int
    concept_indices: list[int]
    text: str
    author_confidence: ClaimConfidence


class _ClaimExtractionResult(BaseModel):
    claims: list[_ExtractedClaim]


class _ExtractedRelation(BaseModel):
    source_index: int
    target_index: int
    relation_type: ClaimRelationType
    rationale: str


class _ClaimLinkResult(BaseModel):
    relations: list[_ExtractedRelation]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _claim_id(source_id: str, chunk_id: str, text: str) -> str:
    payload = "|".join([source_id, chunk_id, _normalize_text(text)])
    return f"claim:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _relation_id(source_claim_id: str, target_claim_id: str, relation_type: ClaimRelationType) -> tuple[str, str, str]:
    """Returns (relation_id, canonical_source, canonical_target). The
    relation type is baked into the id — see ClaimRelation.relation_id's
    docstring for why. "agrees"/"conflicts" are symmetric and get their
    endpoints sorted so re-running linking can never produce both (A,B) and
    (B,A) rows for the same unordered pair; "extends" is directional and
    keeps the order the linker reported."""
    if relation_type in ("agrees", "conflicts"):
        a, b = sorted((source_claim_id, target_claim_id))
        return f"claimrel:{relation_type}:{a}:{b}", a, b
    return f"claimrel:{relation_type}:{source_claim_id}:{target_claim_id}", source_claim_id, target_claim_id


def _format_concepts(concepts: Sequence[Concept]) -> tuple[str, dict[int, str]]:
    index_to_concept_id = {i: c.concept_id for i, c in enumerate(concepts, start=1)}
    text = "\n".join(f"({i}) {c.label}: {c.description}" for i, c in enumerate(concepts, start=1))
    return text, index_to_concept_id


class ClaudeClaimExtractor:
    """ClaimExtractor implementation using Claude (cheap-tier by default —
    see ragcore.model_routing; extraction, not final-answer synthesis)."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_ROUTING.cheap,
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 8192,
        max_retries: int = 5,
    ) -> None:
        if client is None and anthropic is None:
            raise GenerationError(
                "Claude claim extraction requires the 'generation' extra: "
                "install ragcore[generation]"
            ) from _IMPORT_ERROR

        self.model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._client = client or (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def extract(self, *, chunks: Sequence[Chunk], concepts: Sequence[Concept]) -> list[SourceClaim]:
        if not chunks or not concepts:
            return []

        formatted_chunks, index_to_chunk_id = _format_chunks(chunks)
        formatted_concepts, index_to_concept_id = _format_concepts(concepts)
        user_content = (
            f"Chapter chunks:\n{formatted_chunks}\n\nConcepts identified in this chapter:\n{formatted_concepts}"
        )

        parsed = self._call_with_retry(user_content)
        now = _now()

        claims: list[SourceClaim] = []
        for c in parsed.claims:
            chunk_id = index_to_chunk_id.get(c.chunk_index)
            if chunk_id is None:
                continue  # hallucinated chunk index — not a crash-worthy error
            concept_ids = tuple(
                dict.fromkeys(
                    index_to_concept_id[i] for i in c.concept_indices if i in index_to_concept_id
                )
            )
            if not concept_ids:
                continue  # every concept citation was hallucinated — nothing to anchor this claim to

            source_id = next((chunk.meta.source_id for chunk in chunks if chunk.chunk_id == chunk_id), None)
            if source_id is None:
                continue
            claims.append(
                SourceClaim(
                    claim_id=_claim_id(source_id, chunk_id, c.text),
                    concept_ids=concept_ids,
                    source_id=source_id,
                    chunk_id=chunk_id,
                    text=c.text,
                    author_confidence=c.author_confidence,
                    model=self.model,
                    prompt_version=EXTRACTOR_PROMPT_VERSION,
                    created_at=now,
                )
            )
        return claims

    def _call_with_retry(self, user_content: str) -> _ClaimExtractionResult:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.parse(
                    model=self.model,
                    max_tokens=self._max_tokens,
                    system=EXTRACTOR_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                    output_format=_ClaimExtractionResult,
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
                    raise GenerationError(f"{self.model} claim extraction request rejected: {e}") from e

            if attempt < self._max_retries - 1:
                time.sleep(min(2**attempt, 30))

        raise GenerationError(
            f"{self.model} claim extraction failed after {self._max_retries} attempts: {last_err}"
        ) from last_err


class ClaudeClaimLinker:
    """ClaimLinker implementation using Claude (cheap-tier by default —
    see ragcore.model_routing; classification, not final-answer synthesis)."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_ROUTING.cheap,
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 4096,
        max_retries: int = 5,
    ) -> None:
        if client is None and anthropic is None:
            raise GenerationError(
                "Claude claim linking requires the 'generation' extra: "
                "install ragcore[generation]"
            ) from _IMPORT_ERROR

        self.model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._client = client or (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def link(self, claims: Sequence[SourceClaim]) -> list[ClaimRelation]:
        if len(claims) < 2:
            return []

        index_to_claim = {i: c for i, c in enumerate(claims, start=1)}
        formatted = "\n".join(
            f"[{i}] (source={c.source_id[:12]}, author_confidence={c.author_confidence}) {c.text}"
            for i, c in index_to_claim.items()
        )
        parsed = self._call_with_retry(formatted)
        now = _now()

        relations: list[ClaimRelation] = []
        seen_ids: set[str] = set()
        for r in parsed.relations:
            source = index_to_claim.get(r.source_index)
            target = index_to_claim.get(r.target_index)
            if source is None or target is None or source.claim_id == target.claim_id:
                continue  # hallucinated index or a self-pair — dropped silently
            relation_id, canonical_source, canonical_target = _relation_id(
                source.claim_id, target.claim_id, r.relation_type
            )
            if relation_id in seen_ids:
                continue
            seen_ids.add(relation_id)
            relations.append(
                ClaimRelation(
                    relation_id=relation_id,
                    source_claim_id=canonical_source,
                    target_claim_id=canonical_target,
                    relation_type=r.relation_type,
                    rationale=r.rationale,
                    model=self.model,
                    prompt_version=LINKER_PROMPT_VERSION,
                    created_at=now,
                )
            )
        return relations

    def _call_with_retry(self, user_content: str) -> _ClaimLinkResult:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.parse(
                    model=self.model,
                    max_tokens=self._max_tokens,
                    system=LINKER_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                    output_format=_ClaimLinkResult,
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
                    raise GenerationError(f"{self.model} claim linking request rejected: {e}") from e

            if attempt < self._max_retries - 1:
                time.sleep(min(2**attempt, 30))

        raise GenerationError(
            f"{self.model} claim linking failed after {self._max_retries} attempts: {last_err}"
        ) from last_err


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
