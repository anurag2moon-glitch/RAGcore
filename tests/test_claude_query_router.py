"""ClaudeQueryRouter: picks modes from the available list, filters out
anything not actually available, and skips the API call entirely when
there's only one mode to choose from — no API calls, a stub client
stands in for anthropic.Anthropic.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from ragcore.retrieval.routing import ClaudeQueryRouter, _RoutingResult


class _StubResponse:
    def __init__(self, parsed_output: _RoutingResult) -> None:
        self.parsed_output = parsed_output


class _StubMessages:
    def __init__(self, result: _RoutingResult) -> None:
        self._result = result
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return _StubResponse(self._result)


class _StubClient:
    def __init__(self, result: _RoutingResult) -> None:
        self.messages = _StubMessages(result)


def test_route_returns_the_models_chosen_modes() -> None:
    result = _RoutingResult(modes=["vector", "graph_local"], reasoning="needs both")
    client = _StubClient(result)
    router = ClaudeQueryRouter(client=client)

    modes = router.route("who is connected to the captain?", ["vector", "keyword", "graph_local"])

    assert modes == ["vector", "graph_local"]


def test_route_filters_out_modes_not_in_available_modes() -> None:
    result = _RoutingResult(modes=["vector", "made_up_mode"], reasoning="r")
    client = _StubClient(result)
    router = ClaudeQueryRouter(client=client)

    modes = router.route("q", ["vector", "keyword"])

    assert modes == ["vector"]


def test_single_available_mode_skips_the_api_call() -> None:
    client = _StubClient(_RoutingResult(modes=[], reasoning=""))
    router = ClaudeQueryRouter(client=client)

    modes = router.route("q", ["vector"])

    assert modes == ["vector"]
    assert client.messages.calls == []


def test_no_available_modes_returns_empty_without_a_call() -> None:
    client = _StubClient(_RoutingResult(modes=[], reasoning=""))
    router = ClaudeQueryRouter(client=client)

    assert router.route("q", []) == []
    assert client.messages.calls == []


def test_system_prompt_lists_only_the_available_modes() -> None:
    client = _StubClient(_RoutingResult(modes=["vector"], reasoning="r"))
    router = ClaudeQueryRouter(client=client)

    router.route("q", ["vector", "keyword"])

    sent_system = client.messages.calls[0]["system"]
    assert "vector" in sent_system
    assert "keyword" in sent_system
    assert "graph_local" not in sent_system
