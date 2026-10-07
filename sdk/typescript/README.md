# ragcore-client (TypeScript)

TypeScript client SDK for the [ragcore reference server](../../server/).
Zero runtime dependencies — uses Node's built-in `fetch`/`FormData`/`Blob`.

## Install

```
npm install
npm run build
```

## Use

```typescript
import { RagcoreClient } from "ragcore-client";

const client = new RagcoreClient("http://localhost:8000");
await client.ingest("book.md");
for (const chunk of await client.retrieve("what is contextual retrieval?", { k: 5 })) {
  console.log(chunk.score, chunk.text.slice(0, 80));
}
```

### Rules (Phase 10)

```typescript
const rules = await client.listRules("lal_kitab", "my_book");

const result = await client.matchRules("lal_kitab", { "planet.mars.house": 8 });
for (const rule of result.effective) {
  console.log(rule.ruleId, rule.effect);
}

await client.approveRule(rules[0].ruleId, "alice", "checked against the source");
await client.editRule(rules[0].ruleId, "alice", "fix confidence", { confidence: 0.9 });
```

## Test

```
npm test
```

Tests spawn a **real** `server/` FastAPI process (via
`server/tests/run_test_server.py`, using the same in-memory test
doubles `server/tests/test_api.py` does) listening on a real TCP
socket, and exercise this client's actual HTTP requests against it —
unlike the Python SDK's tests, there's no in-process ASGI bridge
available from a Node test runner into a Python app, so this is a real
subprocess + real network call to `127.0.0.1`, not a shortcut. No
Voyage/LanceDB/Anthropic credentials are needed anywhere in the run.

Requires `server/`'s own venv to already exist with its dependencies
installed (`cd ../../server && python3 -m venv .venv && source
.venv/bin/activate && pip install -e .. -e ".[dev]"`) — the test suite
spawns `server/.venv/bin/python`, not whatever `python` happens to be
on `PATH`.
