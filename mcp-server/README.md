# ragcore MCP server

An MCP server exposing one `ragcore.Corpus` as tools for an MCP client
(Claude Desktop, Claude Code, ...): `ingest_file`, `retrieve`, `ask`,
`list_sources`. Single-tenant, no authentication — see
[`docs/adr/0008-api-server-as-sibling-package.md`](../docs/adr/0008-api-server-as-sibling-package.md)
in the repo root for why, and what's deliberately not here yet (the
same reasoning applies as for `server/`, the HTTP sibling).

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e ..              # ragcore itself
pip install -e ".[dev]"        # this MCP server + test tooling
```

## Run

```
export RAGCORE_DB_PATH=./data/corpus.lancedb
python -m mcp_server.server
```

Add to an MCP client's config (e.g. Claude Desktop's `claude_desktop_config.json`
or Claude Code's MCP settings) as a stdio server pointing at that command.

## Test

```
pytest -q
```

No network, no real Voyage/LanceDB credentials needed — `build_server(corpus_provider=...)`
takes an in-memory test Corpus directly, the same dependency-injection
shape `server/`'s `app.dependency_overrides` uses.
