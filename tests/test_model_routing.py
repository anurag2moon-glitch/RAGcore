"""DEFAULT_ROUTING is actually applied: every cheap-tier Claude-backed
component defaults model= to DEFAULT_ROUTING.cheap, every strong-tier one
stays on DEFAULT_ROUTING.strong (docs/ARCHITECTURE.md Phase 5 gap list
item 21). No API calls: only constructor defaults are inspected, via a
stub client so construction never needs anthropic installed+configured.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from ragcore.model_routing import DEFAULT_ROUTING
from tests.fakes import FakeEmbedder


class _StubClient:
    messages = object()


def test_cheap_tier_components_default_to_the_cheap_model() -> None:
    from ragcore.claims.claude import ClaudeClaimExtractor, ClaudeClaimLinker
    from ragcore.concepts.claude_extractor import ClaudeConceptExtractor
    from ragcore.concepts.mapper import EmbeddingConceptMapper, ClaudeConceptMapper
    from ragcore.contextual.claude import ClaudeContextWriter
    from ragcore.conversation.rewriting import HistoryAwareQueryRewriter
    from ragcore.graph.claude_extractor import ClaudeEntityExtractor
    from ragcore.hierarchy.claude import ClaudeHierarchySummarizer
    from ragcore.retrieval.iterative import ClaudeSufficiencyJudge
    from ragcore.retrieval.query_expansion import ClaudeQueryExpander
    from ragcore.retrieval.routing import ClaudeQueryRouter

    client = _StubClient()
    cheap_tier = [
        ClaudeEntityExtractor(client=client),
        ClaudeConceptExtractor(client=client),
        ClaudeClaimExtractor(client=client),
        ClaudeClaimLinker(client=client),
        ClaudeContextWriter(client=client),
        HistoryAwareQueryRewriter(client=client),
        ClaudeQueryExpander(client=client),
        ClaudeQueryRouter(client=client),
        ClaudeSufficiencyJudge(client=client),
        ClaudeHierarchySummarizer(client=client),
        ClaudeConceptMapper(EmbeddingConceptMapper(FakeEmbedder(dim=8)), client=client),
    ]

    for component in cheap_tier:
        assert component.model == DEFAULT_ROUTING.cheap, component


def test_strong_tier_components_default_to_the_strong_model() -> None:
    from ragcore.concepts.communities import ClaudeCommunitySummarizer
    from ragcore.generation.claude import ClaudeGenerator
    from ragcore.verification.claude import ClaudeVerifier

    client = _StubClient()
    strong_tier = [
        ClaudeGenerator(client=client),
        ClaudeVerifier(client=client),
        ClaudeCommunitySummarizer(client=client),
    ]

    for component in strong_tier:
        assert component.model == DEFAULT_ROUTING.strong, component


def test_default_strong_model_is_unchanged_from_before_this_change() -> None:
    """The gap item's own framing: strong-tier defaults must be a no-op
    relative to every component's prior hardcoded default."""
    assert DEFAULT_ROUTING.strong == "claude-opus-5"


def test_an_explicit_model_override_still_wins_over_the_default() -> None:
    from ragcore.graph.claude_extractor import ClaudeEntityExtractor

    extractor = ClaudeEntityExtractor(client=_StubClient(), model="some-other-model")
    assert extractor.model == "some-other-model"
