/**
 * Response types for ragcore-client, mirroring the Python SDK's
 * dataclasses (sdk/python/ragcore_client/types.py) and the reference
 * server's JSON contract (server/app/models.py).
 */

export interface IngestResult {
  sourceIds: string[];
  chunksIngested: number;
  sourcesReplaced: string[];
  errors: [string, string][];
}

export interface SourceInfo {
  sourceId: string;
  path: string;
  title: string | null;
  sourceType: string;
  chunkCount: number;
}

export interface RetrievedChunk {
  chunkId: string;
  text: string;
  score: number;
  sourceId: string;
  sourcePath: string;
  chapterTitle: string | null;
  tenantId: string | null;
  licenseId: string | null;
}

export interface AskResult {
  text: string;
  citations: RetrievedChunk[];
}

export interface RetrieveOptions {
  k?: number;
  filters?: Record<string, unknown>;
  minScore?: number;
}

export interface AskOptions {
  k?: number;
  filters?: Record<string, unknown>;
}

/** Mirrors server/app/rules_models.py's RuleResponse — a projection of
 * ragcore.rules.types.Rule, not a 1:1 mirror, same philosophy as every
 * other type in this file. */
export interface RuleInfo {
  ruleId: string;
  domain: string;
  bookId: string;
  chapter: string | null;
  pageStart: number;
  verbatimSpan: string;
  normalizedStatement: string;
  conditions: Record<string, unknown>;
  effect: Record<string, unknown>;
  remedy: Record<string, unknown> | null;
  ruleType: string;
  confidence: number;
  reviewStatus: string;
  version: number;
}

export interface MatchedRuleInfo {
  rule: RuleInfo;
  isOverridden: boolean;
  overrides: string[];
}

export interface MatchRulesResult {
  matched: MatchedRuleInfo[];
  effective: RuleInfo[];
}

export interface MatchRulesOptions {
  bookId?: string;
  includeUnverified?: boolean;
}

export interface EditRuleOptions {
  effect?: Record<string, unknown>;
  confidence?: number;
  remedy?: Record<string, unknown>;
}

export interface ExtractRulesResult {
  chaptersProcessed: number;
  chaptersSkipped: number;
  candidatesDetected: number;
  rulesExtracted: number;
  rulesStored: number;
  rulesRejectedUnsupported: number;
  errors: [string, string, string][];
}
