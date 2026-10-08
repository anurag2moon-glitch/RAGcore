"""CI integration test (docs/adr/0022) against a *real* running
Compose stack: seed a staging tenant/key, ingest a small sample book,
run a search, and check the citation - the exact flow the deployment
task's own CI section asked for. Run after `docker compose up`, once
`/health/ready` is 200:

    python scripts/ci_integration_test.py [base_url]

Needs RAGCORE_USE_FAKE_MODELS=1 on the api/worker containers (see
app.ci_fakes/app.settings.Settings.ragcore_use_fake_models) - no
VOYAGE_API_KEY/ANTHROPIC_API_KEY secret is assumed to exist in CI,
matching this repo's existing `pytest -m live` exclusion for the same
reason. Exits non-zero (and prints what failed) on any assertion
failure - this script's own exit code is the CI gate.
"""

from __future__ import annotations

import sys
import time

import httpx

SAMPLE_BOOK = b"""# Chapter One: Foxes

The quick brown fox jumps over the lazy dog. Foxes are clever animals
that live in dens and hunt small prey at dusk.

# Chapter Two: Quantum Computing

Quantum computers use qubits instead of classical bits. Qubits can be
entangled, enabling computations impossible for classical machines.
"""

TENANT_ID = "ci-integration-tenant"
API_KEY = "ci-integration-key"


def _wait_for_ready(client: httpx.Client, *, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            resp = client.get("/health/ready")
            if resp.status_code == 200:
                print("api is ready:", resp.json())
                return
            last_error = f"{resp.status_code}: {resp.text}"
        except httpx.RequestError as e:
            last_error = str(e)
        time.sleep(2)
    raise SystemExit(f"api never became ready within {timeout_s}s - last error: {last_error}")


def _wait_for_job(client: httpx.Client, job_id: str, headers: dict, *, timeout_s: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        resp = client.get(f"/jobs/{job_id}", headers=headers)
        resp.raise_for_status()
        body = resp.json()
        if body["status"] == "finished":
            return body
        if body["status"] == "failed":
            raise SystemExit(f"ingestion job failed: {body.get('error')}")
        time.sleep(1)
    raise SystemExit(f"job {job_id} did not finish within {timeout_s}s")


def main() -> None:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    headers = {"X-API-Key": API_KEY}

    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        _wait_for_ready(client)

        print(f"ingesting sample book for tenant {TENANT_ID!r} ...")
        resp = client.post(
            "/ingest",
            files={"file": ("sample_book.md", SAMPLE_BOOK, "text/markdown")},
            headers=headers,
        )
        if resp.status_code != 202:
            raise SystemExit(f"/ingest returned {resp.status_code}: {resp.text}")
        job_id = resp.json()["job_id"]
        print(f"enqueued job {job_id}, waiting for it to finish ...")

        job = _wait_for_job(client, job_id, headers)
        chunks_ingested = job["result"]["chunks_ingested"]
        assert chunks_ingested > 0, f"expected at least one chunk ingested, got {job['result']}"
        print(f"ingested {chunks_ingested} chunk(s)")

        print("running a search for 'foxes' ...")
        resp = client.post("/retrieve", json={"query": "Tell me about foxes and dens", "k": 3}, headers=headers)
        resp.raise_for_status()
        results = resp.json()["results"]
        assert results, "expected at least one retrieved chunk"
        top_text = results[0]["chunk"]["text"]
        assert "fox" in top_text.lower(), f"expected the top result to be about foxes, got: {top_text!r}"
        print(f"top result (score={results[0]['score']:.3f}): {top_text[:60]!r}")

        print("asking a question and checking the citation ...")
        resp = client.post("/ask", json={"query": "Tell me about foxes and dens", "k": 3}, headers=headers)
        resp.raise_for_status()
        answer = resp.json()
        assert answer["citations"], "expected /ask to return at least one citation"
        cited_text = answer["citations"][0]["chunk"]["text"]
        assert "fox" in cited_text.lower(), f"expected the citation to be about foxes, got: {cited_text!r}"
        print(f"answer: {answer['text']!r}")
        print(f"citation: {cited_text[:60]!r}")

    print("CI integration test passed.")


if __name__ == "__main__":
    main()
