# 0019: Structured logging, request IDs, metrics, and tracing

## Status

Accepted.

## Context

The deployment task's "operations basics" section asked for structured
JSON logging, request IDs, a basic metrics endpoint, and OpenTelemetry
tracing wired to a local collector in Compose.

## Decisions

**JSON logging and request IDs are both hand-rolled stdlib `logging`
(`app/logging_config.py`), not a new dependency.** A JSON
`logging.Formatter` subclass and a `logging.Filter` that reads a
`contextvars.ContextVar` are both small enough that a library
(`python-json-logger`, `structlog`, ...) would be more surface area
than the thing it replaces, and stdlib composes cleanly with uvicorn's
own loggers (`configure_logging()` replaces the *root* logger's
handlers, so uvicorn's access/error logs get the same JSON treatment
for free).

**The request id lives in a `contextvars.ContextVar`, not a
thread-local or a plumbed-through parameter.** uvicorn runs an async
event loop where "current thread" doesn't correspond to "current
request" the way it would under a synchronous WSGI server, but
`contextvars` is propagated correctly across `await` points within one
request's own task. Any log call anywhere in the call stack - not just
`request_id_middleware`'s own access-log line - picks up the same id
automatically via a `logging.Filter`, with no `request_id=...`
parameter threading through every function signature in the codebase.

**A caller-supplied `X-Request-ID` is honored, not overwritten.**
Lets an upstream proxy/gateway's own correlation id survive end to end;
one is generated (`uuid4().hex`) only when the header is absent.
Verified directly, not just by inspection: a `TestClient` run confirmed
both the auto-generated and the client-supplied id round-trip onto the
response header and into the resulting JSON log line.

**`/metrics` is a second, independent mechanism from FastAPI's own
OpenTelemetry metrics - deliberately, not a duplicate.** FastAPI 0.142+
ships native `FastAPI(telemetry={...})` OpenTelemetry integration
(tracing/metrics/logs all on by default even with the kwarg omitted
entirely - confirmed directly: a `TestClient` run with no `telemetry=`
passed still created spans and attempted OTLP export). With
`fastapi[opentelemetry]` installed and `OTEL_EXPORTER_OTLP_ENDPOINT`
set, that pushes metrics via OTLP to the collector - useful for a real
tracing backend, but not what "a metrics endpoint" usually means in ops
tooling (a `/metrics` a Prometheus server or a bare `curl` scrapes
directly). `app/metrics.py` adds exactly that: a hand-written
Prometheus text-exposition renderer (no `prometheus_client` dependency
- the format is simple enough not to need one) backed by a small
thread-safe in-memory counter, wired via a second ASGI middleware that
times every request and tags it by method/path-template/status.

**Confirmed, not assumed: FastAPI's telemetry auto-configure degrades
gracefully when the collector is unreachable.** Ran a real request
against a `TestClient` with `OTEL_EXPORTER_OTLP_ENDPOINT` pointed at a
closed port - the export failed with logged retries, but the request
itself still returned `200`. Tracing/metrics failures must never be
able to break a real request, and this confirms they don't.

**The `>=0.142` floor on `fastapi[opentelemetry]` is pinned exactly
because the feature doesn't exist in the dependency's previous
`>=0.110` floor** - installing the old minimum would silently have no
`telemetry()` kwarg and no `FastAPI(telemetry={...})` support at all.

**The collector's own config (`server/otel-collector-config.yaml`)
exports to its own stdout (`debug` exporter), not a real tracing
backend.** Enough to confirm the pipeline is wired end to end this
pass; swapping the exporter for Jaeger/Tempo/etc. later needs no change
to the api/worker containers, which only ever know about the
collector's own OTLP endpoint.

## Consequences

- `/metrics`'s counters are in-process and reset on restart, the same
  limitation already stated plainly for the usage meter and rate
  limiter elsewhere in this server - a real multi-process deployment
  scraping `/metrics` needs every process scraped individually (or a
  shared backend), not aggregated for free.
- The OTLP endpoint must speak `http/protobuf` specifically (FastAPI's
  own auto-configure requirement, confirmed via its own error message) -
  `OTEL_EXPORTER_OTLP_ENDPOINT` must point at the collector's HTTP
  receiver (`:4318`), not its gRPC one (`:4317`).
