# syntax=docker/dockerfile:1.7
#
# Multi-stage build producing three targets from one file - `api`,
# `worker`, `migrate` - sharing 100% of the dependency/code layer (they
# differ only in CMD/healthcheck), rather than three near-identical
# Dockerfiles that could drift apart. Build context must be the repo
# root (not server/), since ragcore/ lives there too:
#
#   docker build -f server/Dockerfile --target api    -t ragcore-api .
#   docker build -f server/Dockerfile --target worker -t ragcore-worker .
#   docker build -f server/Dockerfile --target migrate -t ragcore-migrate .
#
# Pinned dependencies (server/requirements.lock.txt - see its own
# header for how to regenerate it): third-party packages install from
# the lockfile's exact versions, not pyproject.toml's `>=` floors, for
# a reproducible build. `ragcore`/`ragcore-server` themselves (local,
# not on PyPI) install with --no-deps immediately after, since every
# dependency they need is already satisfied by the lockfile.

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# -- builder: resolves/installs everything into site-packages ------------

FROM base AS builder
WORKDIR /build

COPY pyproject.toml ./
COPY ragcore/ ./ragcore/
COPY server/requirements.lock.txt server/pyproject.toml server/README.md ./server/
COPY server/app/ ./server/app/
COPY server/alembic.ini ./server/alembic.ini
COPY server/alembic/ ./server/alembic/
COPY server/scripts/ ./server/scripts/

RUN pip install --upgrade pip \
    && pip install -r server/requirements.lock.txt \
    && pip install --no-deps "./[generation]" \
    && pip install --no-deps "./server"

# -- runtime: small final image, non-root -------------------------------

FROM base AS runtime
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY --from=builder /build/server/app ./app
COPY --from=builder /build/server/alembic.ini ./alembic.ini
COPY --from=builder /build/server/alembic ./alembic
COPY --from=builder /build/server/scripts ./scripts

RUN mkdir -p /app/data && chown -R appuser:appuser /app/data
USER appuser

# -- api -------------------------------------------------------------------

FROM runtime AS api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import sys,urllib.request as u; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/health/ready', timeout=3).status == 200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# -- worker ------------------------------------------------------------

FROM runtime AS worker
# rq's own Worker already implements "finish the in-flight job before
# exiting" on the first SIGTERM (see docs/adr/0017's own research into
# Worker.request_stop) - Docker's default stop sequence (SIGTERM, then
# SIGKILL after --stop-timeout) maps onto that directly, no extra
# signal handling needed here.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD sh -c 'test -f /tmp/rq_worker.pid && kill -0 "$(cat /tmp/rq_worker.pid)"'
CMD ["sh", "-c", "rq worker --url \"$REDIS_URL\" --pid /tmp/rq_worker.pid ingest"]

# -- migrate (one-shot) --------------------------------------------------

FROM runtime AS migrate
CMD ["alembic", "upgrade", "head"]
