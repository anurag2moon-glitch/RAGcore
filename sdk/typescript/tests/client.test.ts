/**
 * RagcoreClient, tested against a REAL server/ FastAPI process over a
 * real TCP socket — spawned via server/tests/run_test_server.py (the
 * same in-memory fakes server/tests/test_api.py uses, imported
 * directly there rather than redefined). Unlike the Python SDK's tests
 * (which can bridge into the ASGI app in-process via FastAPI's
 * TestClient), a Node test runner has no equivalent in-process bridge
 * into a Python ASGI app, so this is a real subprocess + real HTTP,
 * not a shortcut.
 */

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { RagcoreClient } from "../src/client.js";
import { RagcoreClientError } from "../src/errors.js";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const SERVER_DIR = join(__dirname, "..", "..", "..", "server");
const SERVER_PYTHON = join(SERVER_DIR, ".venv", "bin", "python");
const RUN_SCRIPT = join(SERVER_DIR, "tests", "run_test_server.py");
const PORT = 8731;
const BASE_URL = `http://127.0.0.1:${PORT}`;

function spawnServer(extraArgs: string[] = []): Promise<ChildProcessWithoutNullStreams> {
  return new Promise((resolve, reject) => {
    const proc = spawn(SERVER_PYTHON, [RUN_SCRIPT, String(PORT), ...extraArgs], {
      cwd: SERVER_DIR,
    });
    let settled = false;
    proc.stdout.on("data", (data: Buffer) => {
      if (!settled && data.toString().includes("READY")) {
        settled = true;
        resolve(proc);
      }
    });
    proc.on("error", reject);
    proc.on("exit", (code) => {
      if (!settled) reject(new Error(`server process exited early with code ${code}`));
    });
    setTimeout(() => {
      if (!settled) reject(new Error("server did not become READY within 10s"));
    }, 10000);
  });
}

function stopServer(proc: ChildProcessWithoutNullStreams): Promise<void> {
  return new Promise((resolve) => {
    proc.on("exit", () => resolve());
    proc.kill();
  });
}

describe("RagcoreClient against a real server (no generator)", () => {
  let serverProcess: ChildProcessWithoutNullStreams;
  let client: RagcoreClient;
  let tmpDir: string;

  beforeAll(async () => {
    serverProcess = await spawnServer();
    client = new RagcoreClient(BASE_URL);
    tmpDir = await mkdtemp(join(tmpdir(), "ragcore-ts-sdk-"));
  }, 15000);

  afterAll(async () => {
    await stopServer(serverProcess);
    await rm(tmpDir, { recursive: true, force: true });
  });

  it("reports healthy", async () => {
    expect(await client.health()).toBe(true);
  });

  it("ingests a file, then lists it as a source, then retrieves from it", async () => {
    const bookPath = join(tmpDir, "book.md");
    await writeFile(bookPath, "# Chapter 1\n\nSome content about chunking and embeddings.\n");

    const report = await client.ingest(bookPath);
    expect(report.chunksIngested).toBeGreaterThan(0);

    const sources = await client.sources();
    expect(sources.some((s) => s.sourceId === report.sourceIds[0])).toBe(true);

    const results = await client.retrieve("chunking", { k: 5 });
    expect(results.length).toBeGreaterThan(0);
    expect(results.every((r) => typeof r.chunkId === "string")).toBe(true);
  });

  it("raises RagcoreClientError for /ask with no generator configured", async () => {
    await expect(client.ask("anything")).rejects.toThrow(RagcoreClientError);
    try {
      await client.ask("anything");
      expect.unreachable("expected ask() to throw");
    } catch (err) {
      expect(err).toBeInstanceOf(RagcoreClientError);
      expect((err as RagcoreClientError).statusCode).toBe(400);
    }
  });

  it("deletes a source and removes its chunks", async () => {
    const bookPath = join(tmpDir, "book2.md");
    await writeFile(bookPath, "# Chapter 1\n\nSome other content.\n");
    const report = await client.ingest(bookPath);

    const deleted = await client.deleteSource(report.sourceIds[0]);

    expect(deleted).toBeGreaterThan(0);
    const sources = await client.sources();
    expect(sources.some((s) => s.sourceId === report.sourceIds[0])).toBe(false);
  });
});

describe("RagcoreClient against a real server (with a generator)", () => {
  let serverProcess: ChildProcessWithoutNullStreams;
  let client: RagcoreClient;
  let tmpDir: string;

  beforeAll(async () => {
    serverProcess = await spawnServer(["--with-generator"]);
    client = new RagcoreClient(BASE_URL);
    tmpDir = await mkdtemp(join(tmpdir(), "ragcore-ts-sdk-"));
  }, 15000);

  afterAll(async () => {
    await stopServer(serverProcess);
    await rm(tmpDir, { recursive: true, force: true });
  });

  it("asks a question and gets a generated answer with citations", async () => {
    const bookPath = join(tmpDir, "book.md");
    await writeFile(bookPath, "# Chapter 1\n\nSome content about chunking.\n");
    await client.ingest(bookPath);

    const result = await client.ask("chunking");

    expect(result.text).toContain("answer to");
    expect(result.citations.length).toBeGreaterThan(0);
  });
});

describe("RagcoreClient rule methods against a real server", () => {
  let serverProcess: ChildProcessWithoutNullStreams;
  let client: RagcoreClient;

  beforeAll(async () => {
    serverProcess = await spawnServer(["--with-sample-rules"]);
    client = new RagcoreClient(BASE_URL);
  }, 15000);

  afterAll(async () => {
    await stopServer(serverProcess);
  });

  it("lists rules for a domain", async () => {
    const rules = await client.listRules("test");
    expect(rules.map((r) => r.ruleId).sort()).toEqual([
      "rule:sample_pending",
      "rule:sample_verified",
    ]);
  });

  it("matches rules against a fact set, excluding unverified by default", async () => {
    const result = await client.matchRules("test", { "planet.mars.house": 8 });
    expect(result.matched.map((m) => m.rule.ruleId)).toEqual(["rule:sample_verified"]);
    expect(result.effective.map((r) => r.ruleId)).toEqual(["rule:sample_verified"]);
  });

  it("includes unverified rules when asked", async () => {
    const result = await client.matchRules(
      "test",
      { "planet.mars.house": 8 },
      { includeUnverified: true },
    );
    expect(result.matched.map((m) => m.rule.ruleId).sort()).toEqual([
      "rule:sample_pending",
      "rule:sample_verified",
    ]);
  });

  it("approves a pending rule", async () => {
    const approved = await client.approveRule("rule:sample_pending", "alice", "looks right");
    expect(approved.reviewStatus).toBe("verified");
  });

  it("edits a rule, creating a new version", async () => {
    const edited = await client.editRule("rule:sample_verified", "carol", "bump confidence", {
      confidence: 0.99,
    });
    expect(edited.ruleId).not.toBe("rule:sample_verified");
    expect(edited.confidence).toBe(0.99);
    expect(edited.version).toBe(2);
  });

  it("raises RagcoreClientError for an unknown rule", async () => {
    try {
      await client.approveRule("rule:nonexistent", "alice");
      expect.unreachable("expected approveRule() to throw");
    } catch (err) {
      expect(err).toBeInstanceOf(RagcoreClientError);
      expect((err as RagcoreClientError).statusCode).toBe(404);
    }
  });
});
