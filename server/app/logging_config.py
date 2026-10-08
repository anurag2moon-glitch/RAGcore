"""Structured JSON logging + request IDs (the deployment request's own
ask). Stdlib `logging` only — no new dependency, since a JSON formatter
is a small enough `logging.Formatter` subclass that a library would be
more surface area than the thing it replaces.

The request id is a `contextvars.ContextVar`, not a thread-local: this
server runs under `uvicorn`'s async event loop, where "current thread"
doesn't correspond to "current request" the way it would under a
synchronous, one-thread-per-request WSGI server. A `logging.Filter`
reads the contextvar on every record so any log call anywhere in the
request's call stack - not just the access-log line `request_id_middleware`
itself emits - picks up the same id for free, with no `request_id=...`
plumbed through every function signature.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid

_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ragcore_request_id", default=None
)


def get_current_request_id() -> str | None:
    return _request_id_var.get()


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id is not None:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str) -> None:
    """Call once, at startup - replaces the root logger's handlers so
    every logger in the process (uvicorn's own included) emits one JSON
    object per line to stdout, tagged with the in-flight request's id."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


async def request_id_middleware(request, call_next):
    incoming = request.headers.get("x-request-id")
    request_id = incoming or uuid.uuid4().hex
    token = _request_id_var.set(request_id)
    start = time.monotonic()
    try:
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        # Logged while the contextvar is still set, so the access line
        # itself picks up request_id via the filter - resetting first
        # would make this one line log with no request_id at all.
        logging.getLogger("ragcore_server.access").info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
    finally:
        _request_id_var.reset(token)
