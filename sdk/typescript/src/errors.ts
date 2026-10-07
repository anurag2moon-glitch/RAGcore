/** Thrown when the reference server returns a non-2xx response. */
export class RagcoreClientError extends Error {
  readonly statusCode: number;
  readonly detail: string;

  constructor(statusCode: number, detail: string) {
    super(`server returned ${statusCode}: ${detail}`);
    this.name = "RagcoreClientError";
    this.statusCode = statusCode;
    this.detail = detail;
  }
}
