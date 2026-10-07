# ragcore-client

Python client SDK for the [ragcore reference server](../../server/).

## Install

```
pip install -e .
```

## Use

```python
from ragcore_client import RagcoreClient

with RagcoreClient("http://localhost:8000") as client:
    client.ingest("book.md")
    for chunk in client.retrieve("what is contextual retrieval?", k=5):
        print(chunk.score, chunk.text[:80])
```

### Rules (Phase 10)

```python
with RagcoreClient("http://localhost:8000") as client:
    rules = client.list_rules("lal_kitab", book_id="my_book")

    result = client.match_rules("lal_kitab", {"planet.mars.house": 8})
    for rule in result.effective:
        print(rule.rule_id, rule.effect)

    client.approve_rule(rules[0].rule_id, reviewer="alice", reason="checked against the source")
    client.edit_rule(rules[0].rule_id, reviewer="alice", reason="fix confidence", confidence=0.9)
```

## Test

Tests exercise this client against the real `server/` FastAPI app
object in-process (via `httpx`'s `ASGITransport`), not a real running
server or network call — see `tests/test_client.py`. From this
directory:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]" -e ../../
pytest -q
```
