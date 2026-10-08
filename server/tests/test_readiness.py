"""GET /health, /health/live, /health/ready: liveness never fails;
readiness only checks whichever dependency is actually configured,
and an unconfigured one never counts as a failure.
"""

from __future__ import annotations

import pytest
from app.main import app
from app.settings import Settings
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def test_health_is_always_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_live_is_always_ok(client: TestClient) -> None:
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readiness_with_nothing_configured_is_trivially_ready(client: TestClient) -> None:
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready", "checks": {}}


def test_readiness_reports_an_unreachable_redis(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.readiness.get_settings",
        lambda: Settings(redis_url="redis://localhost:59999/0"),
    )

    resp = client.get("/health/ready")

    assert resp.status_code == 503
    body = resp.json()["detail"]
    assert body["status"] == "not ready"
    assert "error" in body["checks"]["redis"]


def test_readiness_reports_a_reachable_redis(client: TestClient, monkeypatch) -> None:
    import fakeredis
    import redis

    monkeypatch.setattr(redis, "from_url", fakeredis.FakeStrictRedis.from_url)
    monkeypatch.setattr(
        "app.readiness.get_settings", lambda: Settings(redis_url="redis://irrelevant:6379/0")
    )

    resp = client.get("/health/ready")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ready", "checks": {"redis": "ok"}}
