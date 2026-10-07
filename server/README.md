# ragcore reference server

A minimal FastAPI service exposing one `ragcore.Corpus` over HTTP:
ingest a file, retrieve, ask, list/delete sources. Single-tenant, no
authentication — see
[`docs/adr/0008-api-server-as-sibling-package.md`](../docs/adr/0008-api-server-as-sibling-package.md)
in the repo root for why, and what's deliberately not here yet.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e ..              # ragcore itself
pip install -e ".[dev]"        # this server + test tooling
```

## Run

```
export RAGCORE_DB_PATH=./data/corpus.lancedb
uvicorn app.main:app --reload
```

Visit `/docs` for the interactive OpenAPI UI (generated automatically
by FastAPI from `app/main.py`'s routes and `app/models.py`'s Pydantic
models — no separate authoring step).

## Test

```
pytest -q
```

No network, no real Voyage/LanceDB credentials needed — `app.corpus_provider.get_corpus`
is overridden with in-memory test doubles via FastAPI's
`app.dependency_overrides`.

## Endpoints

| Method | Path | Body | Notes |
|---|---|---|---|
| `GET` | `/health` | — | liveness check |
| `POST` | `/ingest` | multipart `file` | dispatches by the uploaded filename's extension |
| `POST` | `/retrieve` | `{query, k, filters, min_score}` | |
| `POST` | `/ask` | `{query, k, filters}` | `400` if the corpus has no `Generator` configured |
| `GET` | `/sources` | — | |
| `DELETE` | `/sources/{source_id}` | — | |
