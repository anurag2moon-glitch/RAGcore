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
