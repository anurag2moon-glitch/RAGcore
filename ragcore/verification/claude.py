"""ClaudeVerifier: post-generation citation verification, and
VerifyingGenerator, the Generator decorator that wires it in.

Two genuinely separate messages.parse calls, not a "please cite your
sources" instruction folded into the generation prompt: (1) decompose
Answer.text into atomic claims, (2) check every claim against the numbered
context items (reusing the same `[n]` scheme
ragcore.generation.claude.ClaudeGenerator already shows the model) and
report which support it.

"supported_by_synthesis" is never trusted to the model's own judgment — it
is derived mechanically here from provenance: a claim is
supported_by_synthesis iff every context item supporting it is a synthetic
community-summary chunk (see ragcore.retrieval.concept and ChunkOrigin in
ragcore.types). The model's actual job is only "which numbered items support
this claim"; status is computed afterward from what it points at.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Sequence

from pydantic import BaseModel

from ragcore.errors import GenerationError
from ragcore.generation.base import Answer, Generator
from ragcore.graph.claude_extractor import _format_chunks
from ragcore.types import Retrieved
from ragcore.verification.base import Verifier
from ragcore.verification.types import ClaimCheck, ClaimStatus, Verification

try:
    import anthropic
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    anthropic = None
else:
    _IMPORT_ERROR = None

DECOMPOSE_SYSTEM_PROMPT = (
    "You split an answer into its individual factual claims, for a "
    "groundedness check. Each claim should be a single self-contained "
    "assertion, quoted verbatim (or as close to verbatim as possible) from "
    "the given text — do not paraphrase or add anything not in the "
    "original. Skip purely structural text (headers, \"in summary\", "
    "transitions) that asserts nothing. If the text makes no checkable "
    "claims at all, return an empty list."
)

CHECK_SYSTEM_PROMPT = (
    "You check whether each of a list of claims is actually supported by a "
    "set of numbered source excerpts. You are given the claims (each "
    "numbered) and the excerpts (each numbered in brackets, like [1]). For "
    "each claim, list the bracketed number(s) of every excerpt that "
    "genuinely supports it (e.g. 1, 3 — not the text \"[1]\", just the "
    "number) — an empty list if none do. Be strict: a claim is only "
    "supported if the excerpt(s) actually state or clearly imply it, not "
    "just mention related words."
)


class _ClaimList(BaseModel):
    claims: list[str]


class _CheckedClaim(BaseModel):
    claim_index: int
    supporting_indices: list[int]


class _CheckResult(BaseModel):
    checks: list[_CheckedClaim]


def _is_synthesis(chunk_id: str, citations: Sequence[Retrieved]) -> bool:
    for r in citations:
        if r.chunk.chunk_id == chunk_id:
            return r.chunk.meta.extra.get("ragcore_kind") == "community_summary"
    return False


class ClaudeVerifier:
    """Verifier implementation using Claude (claude-opus-5 by default)."""

    def __init__(
        self,
        *,
        model: str = "claude-opus-5",
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 4096,
        max_retries: int = 5,
    ) -> None:
        if client is None and anthropic is None:
            raise GenerationError(
                "Claude verification requires the 'generation' extra: "
                "install ragcore[generation]"
            ) from _IMPORT_ERROR

        self.model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._client = client or (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def verify(self, query: str, answer: Answer) -> Verification:
        if not answer.text.strip() or not answer.citations:
            return Verification(checks=(), model=self.model)

        claims = self._decompose(answer.text)
        if not claims:
            return Verification(checks=(), model=self.model)

        context_chunks = [r.chunk for r in answer.citations]
        formatted_context, index_to_chunk_id = _format_chunks(context_chunks)
        numbered_claims = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))
        user_content = f"Claims:\n{numbered_claims}\n\nSource excerpts:\n{formatted_context}"
        result: _CheckResult = self._call_with_retry(CHECK_SYSTEM_PROMPT, user_content, _CheckResult)

        checks_by_index = {c.claim_index: c for c in result.checks}
        checks: list[ClaimCheck] = []
        for i, claim in enumerate(claims, start=1):
            checked = checks_by_index.get(i)
            support = tuple(
                index_to_chunk_id[idx]
                for idx in (checked.supporting_indices if checked else [])
                if idx in index_to_chunk_id
            )
            if not support:
                status: ClaimStatus = "unsupported"
            elif all(_is_synthesis(cid, answer.citations) for cid in support):
                status = "supported_by_synthesis"
            else:
                status = "supported"
            checks.append(ClaimCheck(claim=claim, status=status, support=support))

        return Verification(checks=tuple(checks), model=self.model)

    def _decompose(self, text: str) -> list[str]:
        result: _ClaimList = self._call_with_retry(DECOMPOSE_SYSTEM_PROMPT, text, _ClaimList)
        return result.claims

    def _call_with_retry(self, system_prompt: str, user_content: str, output_format: type) -> Any:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.parse(
                    model=self.model,
                    max_tokens=self._max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_content}],
                    output_format=output_format,
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
                    raise GenerationError(f"{self.model} verification request rejected: {e}") from e

            if attempt < self._max_retries - 1:
                time.sleep(min(2**attempt, 30))

        raise GenerationError(
            f"{self.model} verification failed after {self._max_retries} attempts: {last_err}"
        ) from last_err


class VerifyingGenerator:
    """Generator decorator: calls inner.generate(), then verifier.verify()
    on the result, attaching Verification to the returned Answer.
    Structurally satisfies Generator, so it composes anywhere a Generator is
    expected — including nesting under another decorator.

    Wiring is one line, the same seam GraphRetriever uses for Retriever:
        corpus.generator = VerifyingGenerator(corpus.generator, verifier)
    A consumer who never does this sees no change at all — Answer.verification
    stays None, exactly as ClaudeGenerator/FakeGenerator already leave it.
    """

    def __init__(self, inner: Generator, verifier: Verifier) -> None:
        self._inner = inner
        self._verifier = verifier
        self.model = inner.model

    def generate(self, query: str, context: Sequence[Retrieved]) -> Answer:
        answer = self._inner.generate(query, context)
        verification = self._verifier.verify(query, answer)
        return dataclasses.replace(answer, verification=verification)
