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
