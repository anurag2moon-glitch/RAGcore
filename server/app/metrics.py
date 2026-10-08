"""A basic `/metrics` endpoint (the deployment request's own ask) in
Prometheus's plain text exposition format — hand-written, no new
dependency, since the format itself is simple enough (a handful of
`# TYPE`/`# HELP` lines plus one line per label combination) that
pulling in `prometheus_client` just to generate it would be a bigger
dependency than the thing it generates.

Deliberately separate from `fastapi[opentelemetry]`'s own metrics
(exported via OTLP to the collector, server/otel-collector-config.yaml)
— those are for a real tracing/metrics backend; this is the plain-text
`/metrics` a Prometheus server (or anyone with `curl`) scrapes
directly, the more common meaning of "a metrics endpoint" in ops
tooling. The two are complementary, not a duplicate of each other.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict


class RequestMetrics:
    """Thread-safe, in-memory counters — same reasoning as every other
    in-memory reference implementation in this server (usage meter,
    rate limiter): a real multi-process deployment scraping this
    needs a shared backend, stated plainly, not silently assumed."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_count: dict[tuple[str, str, int], int] = defaultdict(int)
        self._latency_sum_seconds: dict[tuple[str, str], float] = defaultdict(float)

    def record(self, method: str, path: str, status_code: int, duration_seconds: float) -> None:
        with self._lock:
            self._request_count[(method, path, status_code)] += 1
            self._latency_sum_seconds[(method, path)] += duration_seconds

    def render_prometheus_text(self) -> str:
        with self._lock:
            counts = dict(self._request_count)
            latencies = dict(self._latency_sum_seconds)

        lines = [
            "# HELP ragcore_http_requests_total Total HTTP requests handled.",
            "# TYPE ragcore_http_requests_total counter",
        ]
        for (method, path, status_code), count in sorted(counts.items()):
            lines.append(
                f'ragcore_http_requests_total{{method="{method}",path="{path}",status="{status_code}"}} {count}'
            )

        lines.append("# HELP ragcore_http_request_duration_seconds_sum Total time spent handling requests.")
        lines.append("# TYPE ragcore_http_request_duration_seconds_sum counter")
        for (method, path), total_seconds in sorted(latencies.items()):
            lines.append(
                f'ragcore_http_request_duration_seconds_sum{{method="{method}",path="{path}"}} {total_seconds}'
            )

        return "\n".join(lines) + "\n"


_metrics = RequestMetrics()


def get_request_metrics() -> RequestMetrics:
    return _metrics


async def metrics_middleware(request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration = time.monotonic() - start
    route = request.scope.get("route")
    path_template = route.path if route is not None else request.url.path
    get_request_metrics().record(request.method, path_template, response.status_code, duration)
    return response
