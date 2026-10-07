# ragcore MCP server

An MCP server exposing one `ragcore.Corpus` and its Phase 10 rule base
as tools for an MCP client (Claude Desktop, Claude Code, ...):
`ingest_file`, `retrieve`, `ask`, `list_sources`, `list_rules`,
`match_rules`, `review_rule`, `extract_rules`. Single-tenant, no
authentication — see
[`docs/adr/0008-api-server-as-sibling-package.md`](../docs/adr/0008-api-server-as-sibling-package.md)
in the repo root for why, and what's deliberately not here yet (the
same reasoning applies as for `server/`, the HTTP sibling).

The rule tools' domain pack registry is empty by default (no hard
dependency on any specific domain pack, e.g. Lal Kitab) — pass
`domain_pack_registry_provider=...` to `build_server()` to register
one. `extract_rules` also needs `rule_extraction_pipeline_provider=...`
(a real `ClaudeRuleCandidateDetector`/`ClaudeRuleExtractor`/
`ClaudeRuleVerifier` triple) — there's no inert default for "how do I
extract rules," so it raises until configured.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e "..[generation]"   # ragcore itself, with the rules layer's pydantic dependency
pip install -e ".[dev]"           # this MCP server + test tooling
```

## Run

```
export RAGCORE_DB_PATH=./data/corpus.lancedb
export RAGCORE_RULES_DB_PATH=./data/rules.sqlite3
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
