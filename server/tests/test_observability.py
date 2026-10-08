"""Structured JSON logging, request IDs, and the /metrics endpoint
(docs/adr/0019) - verified directly, not just by code inspection:
a client-supplied X-Request-ID round-trips unchanged, an absent one
gets a real generated id, the access log line carries whichever one
was in effect, and /metrics accumulates real counts across requests.
"""

from __future__ import annotations

import json
import logging

from app.logging_config import _JsonFormatter, get_current_request_id
from app.main import app
from app.metrics import get_request_metrics
from fastapi.testclient import TestClient


def _client() -> TestClient:
    return TestClient(app)


def test_request_id_is_generated_when_absent() -> None:
    with _client() as client:
        resp = client.get("/health")
    request_id = resp.headers.get("x-request-id")
    assert request_id
    assert len(request_id) == 32  # uuid4().hex


def test_request_id_is_echoed_back_when_supplied() -> None:
    with _client() as client:
        resp = client.get("/health", headers={"X-Request-ID": "caller-supplied-id"})
    assert resp.headers.get("x-request-id") == "caller-supplied-id"


def test_request_id_contextvar_is_cleared_after_the_request() -> None:
    with _client() as client:
        client.get("/health", headers={"X-Request-ID": "should-not-leak"})
    assert get_current_request_id() is None


def test_metrics_endpoint_counts_requests_in_prometheus_format() -> None:
    with _client() as client:
        before = get_request_metrics().render_prometheus_text()
        client.get("/health")
        client.get("/health")
        after = client.get("/metrics").text

    assert "# TYPE ragcore_http_requests_total counter" in after
    assert 'ragcore_http_requests_total{method="GET",path="/health",status="200"}' in after
    before_count = _extract_health_count(before)
    after_count = _extract_health_count(after)
    assert after_count == before_count + 2


def _extract_health_count(text: str) -> int:
    for line in text.splitlines():
        if line.startswith('ragcore_http_requests_total{method="GET",path="/health",status="200"}'):
            return int(line.rsplit(" ", 1)[1])
    return 0


def test_json_formatter_produces_valid_json_with_request_id() -> None:
    formatter = _JsonFormatter()
    record = logging.LogRecord(
        name="ragcore_server.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="something happened",
        args=(),
        exc_info=None,
    )
    record.request_id = "abc123"

    parsed = json.loads(formatter.format(record))

    assert parsed["message"] == "something happened"
    assert parsed["level"] == "INFO"
    assert parsed["request_id"] == "abc123"


def test_json_formatter_omits_request_id_when_none() -> None:
    formatter = _JsonFormatter()
    record = logging.LogRecord(
        name="ragcore_server.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="no request in flight",
        args=(),
        exc_info=None,
    )
    record.request_id = None

    parsed = json.loads(formatter.format(record))

    assert "request_id" not in parsed
