/**
 * RagcoreClient — a thin, typed wrapper over the reference server's
 * HTTP API (server/app/main.py). See server/README.md for the endpoint
 * table this mirrors, and sdk/python/ragcore_client/client.py for the
 * equivalent Python SDK this one is kept in sync with.
 *
 * The server's JSON responses use snake_case field names
 * (server/app/models.py's Pydantic aliases); this client maps them to
 * the camelCase types in ./types.ts at the boundary, once, here —
 * callers never see a snake_case key.
 */

import { readFile } from "node:fs/promises";
import { basename } from "node:path";
import { RagcoreClientError } from "./errors.js";
import type {
  AskOptions,
  AskResult,
  EditRuleOptions,
  ExtractRulesResult,
  IngestResult,
  MatchedRuleInfo,
  MatchRulesOptions,
  MatchRulesResult,
  RetrievedChunk,
  RetrieveOptions,
  RuleInfo,
  SourceInfo,
} from "./types.js";

function chunkFromJson(d: any): RetrievedChunk {
  return {
    chunkId: d.chunk_id,
    text: d.text,
    score: d.score,
    sourceId: d.source_id,
    sourcePath: d.source_path,
    chapterTitle: d.chapter_title ?? null,
    tenantId: d.tenant_id ?? null,
    licenseId: d.license_id ?? null,
  };
}

function ruleFromJson(d: any): RuleInfo {
  return {
    ruleId: d.rule_id,
    domain: d.domain,
    bookId: d.book_id,
    chapter: d.chapter ?? null,
    pageStart: d.page_start,
    verbatimSpan: d.verbatim_span,
    normalizedStatement: d.normalized_statement,
    conditions: d.conditions,
    effect: d.effect,
    remedy: d.remedy ?? null,
    ruleType: d.rule_type,
    confidence: d.confidence,
    reviewStatus: d.review_status,
    version: d.version,
  };
}

function matchedRuleFromJson(d: any): MatchedRuleInfo {
  return { rule: ruleFromJson(d.rule), isOverridden: d.is_overridden, overrides: d.overrides };
}

export class RagcoreClient {
  private readonly baseUrl: string;
  private readonly fetchFn: typeof fetch;

  constructor(baseUrl: string, options: { fetchFn?: typeof fetch } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.fetchFn = options.fetchFn ?? fetch;
  }

  private async request(method: string, path: string, init: RequestInit = {}): Promise<any> {
    const resp = await this.fetchFn(`${this.baseUrl}${path}`, { method, ...init });
    if (!resp.ok) {
      let detail = await resp.text();
      try {
        const body = JSON.parse(detail);
        detail = body.detail ?? detail;
      } catch {
        // not JSON — keep the raw text as the detail
      }
      throw new RagcoreClientError(resp.status, detail);
    }
    return resp.json();
  }

  async health(): Promise<boolean> {
    const body = await this.request("GET", "/health");
    return body.status === "ok";
  }

  async ingest(path: string): Promise<IngestResult> {
    const content = await readFile(path);
    const form = new FormData();
    form.append("file", new Blob([content]), basename(path));
    const body = await this.request("POST", "/ingest", { body: form });
    return {
      sourceIds: body.source_ids,
      chunksIngested: body.chunks_ingested,
      sourcesReplaced: body.sources_replaced,
      errors: body.errors,
    };
  }

  async retrieve(query: string, options: RetrieveOptions = {}): Promise<RetrievedChunk[]> {
    const body = await this.request("POST", "/retrieve", {
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        k: options.k ?? 10,
        filters: options.filters ?? null,
        min_score: options.minScore ?? null,
      }),
    });
    return body.results.map(chunkFromJson);
  }

  async ask(query: string, options: AskOptions = {}): Promise<AskResult> {
    const body = await this.request("POST", "/ask", {
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k: options.k ?? 10, filters: options.filters ?? null }),
    });
    return { text: body.text, citations: body.citations.map(chunkFromJson) };
  }

  async sources(): Promise<SourceInfo[]> {
    const body = await this.request("GET", "/sources");
    return body.map((s: any) => ({
      sourceId: s.source_id,
      path: s.path,
      title: s.title ?? null,
      sourceType: s.source_type,
      chunkCount: s.chunk_count,
    }));
  }

  async deleteSource(sourceId: string): Promise<number> {
    const body = await this.request("DELETE", `/sources/${encodeURIComponent(sourceId)}`);
    return body.chunks_deleted;
  }

  // -- Phase 10: rules (server/app/rules_routes.py) --------------------

  async listRules(domain: string, bookId?: string): Promise<RuleInfo[]> {
    const params = new URLSearchParams({ domain });
    if (bookId !== undefined) params.set("book_id", bookId);
    const body = await this.request("GET", `/rules?${params.toString()}`);
    return body.map(ruleFromJson);
  }

  async matchRules(
    domain: string,
    facts: Record<string, unknown>,
    options: MatchRulesOptions = {},
  ): Promise<MatchRulesResult> {
    const body = await this.request("POST", "/rules/match", {
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        domain,
        facts,
        book_id: options.bookId ?? null,
        include_unverified: options.includeUnverified ?? false,
      }),
    });
    return {
      matched: body.matched.map(matchedRuleFromJson),
      effective: body.effective.map(ruleFromJson),
    };
  }

  async approveRule(ruleId: string, reviewer: string, reason = ""): Promise<RuleInfo> {
    const body = await this.request("POST", `/rules/${encodeURIComponent(ruleId)}/approve`, {
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reviewer, reason }),
    });
    return ruleFromJson(body);
  }

  async rejectRule(ruleId: string, reviewer: string, reason: string): Promise<RuleInfo> {
    const body = await this.request("POST", `/rules/${encodeURIComponent(ruleId)}/reject`, {
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reviewer, reason }),
    });
    return ruleFromJson(body);
  }

  async editRule(
    ruleId: string,
    reviewer: string,
    reason: string,
    options: EditRuleOptions = {},
  ): Promise<RuleInfo> {
    const body = await this.request("POST", `/rules/${encodeURIComponent(ruleId)}/edit`, {
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reviewer,
        reason,
        effect: options.effect ?? null,
        confidence: options.confidence ?? null,
        remedy: options.remedy ?? null,
      }),
    });
    return ruleFromJson(body);
  }

  async extractRules(domain: string, sourceId?: string): Promise<ExtractRulesResult> {
    const params = new URLSearchParams({ domain });
    if (sourceId !== undefined) params.set("source_id", sourceId);
    const body = await this.request("POST", `/rules/extract?${params.toString()}`);
    return {
      chaptersProcessed: body.chapters_processed,
      chaptersSkipped: body.chapters_skipped,
      candidatesDetected: body.candidates_detected,
      rulesExtracted: body.rules_extracted,
      rulesStored: body.rules_stored,
      rulesRejectedUnsupported: body.rules_rejected_unsupported,
      errors: body.errors,
      totalTokensUsed: body.total_tokens_used,
    };
  }
}
