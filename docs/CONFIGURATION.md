# Configuration reference

Phase 9 deliverable. `ragcore` itself reads **zero** environment
variables directly — confirmed by `grep -rn "os.environ\|getenv"
ragcore/`, which returns nothing. Every Claude-/Voyage-/OpenAI-backed
component takes an optional `api_key: str | None = None` constructor
argument and, when omitted, falls through to the underlying vendor
SDK's *own* client construction (`anthropic.Anthropic()`,
`voyageai.Client()`, `openai.OpenAI()`) — which is what actually reads
the environment variables below. This is a direct consequence of
[ADR 0009](adr/0009-library-not-application.md): a library that read
its own environment would be making a policy decision a consumer
should make instead.

## Environment variables

| Variable | Read by | Required for |
|---|---|---|
| `VOYAGE_API_KEY` | `voyageai`'s own SDK (not `ragcore`) | Any ingestion or `retrieve()`/`ask()` call using the default `VoyageContextEmbedder`, and `VoyageReranker` cross-encoder reranking. |
| `ANTHROPIC_API_KEY` | `anthropic`'s own SDK (not `ragcore`) | `Corpus.ask()`/`ClaudeGenerator`, plus every opt-in Claude-backed layer: entity/concept/claim extraction, community/hierarchy summarization, `ClaudeConceptMapper`, `ClaudeVerifier`, contextual-embedding blurb writing, `ClaudeSufficiencyJudge`, `HistoryAwareQueryRewriter`, `ClaudeGoldSetGenerator`. |
| `OPENAI_API_KEY` | `openai`'s own SDK (not `ragcore`) | Only if you construct `OpenAIEmbedder` instead of the default `VoyageContextEmbedder`. |
| `RAGCORE_DB_PATH` | `server/app/corpus_provider.py`, `mcp-server/mcp_server/corpus_provider.py` | Running either reference deployment (`server/`, `mcp-server/`) — the on-disk LanceDB path each one opens via `Corpus.open(path)` at first request. **Not read by `ragcore` itself** — each sibling package reads it independently (deliberately duplicated rather than shared; see `mcp-server/mcp_server/corpus_provider.py`'s own docstring). |
| `RAGCORE_TEST_POSTGRES_URL` | `tests/` only (`pyproject.toml`'s `postgres` pytest marker) | Running the `postgres`-marked test suite against a real Postgres server. Unset by default — those tests are excluded from the default `pytest` run and skip cleanly rather than failing when this isn't set. See [ADR 0005](adr/0005-postgres-store-untested-against-live-server.md). |

Every one of the three API-key variables can be bypassed entirely by
passing `api_key=...` directly to the relevant constructor instead
(`VoyageContextEmbedder(api_key=...)`, `ClaudeGenerator(api_key=...)`,
etc.) — useful for per-tenant credentials in a multi-tenant deployment,
since environment variables are process-wide.

## Installation extras (`pyproject.toml`)

`pip install ragcore` alone gets you the core pipeline over plain-text
input (`tiktoken` is the only hard dependency). Everything else is opt-in:

| Extra | Adds | Needed for |
|---|---|---|
| `pdf` | `pymupdf` | `PdfLoader`. |
| `ocr` | `pytesseract`, `pillow` | OCR fallback for image-only PDF pages. |
| `epub` | `ebooklib`, `selectolax` | `EpubLoader`. |
| `html` | `selectolax` | `HtmlLoader`. |
| `docx` | `python-docx` | `DocxLoader`. |
| `voyage` | `voyageai` | `VoyageContextEmbedder` (the default embedder), `VoyageReranker`. |
| `openai` | `openai` | `OpenAIEmbedder`. |
| `lancedb` | `lancedb`, `pyarrow` | `LanceDBStore` (the default vector store). |
| `kuzu` | `kuzu` | `KuzuGraphStore` (the alternative to `SqliteGraphStore`, which needs no extra). |
| `s3` | `boto3` | `S3RawFileStore`. |
| `postgres` | `psycopg[binary]` | `PostgresIngestCheckpointStore`. |
| `generation` | `anthropic`, `pydantic` | Every Claude-backed component — generation, verification, graph/concept/claim extraction, summarization, routing/judging, query rewriting. |
| `concepts` | `networkx` | Community detection (`ragcore.concepts.communities`). |
| `all` | every extra above | A typical full-stack dev install — what `pip install -e ".[all,dev]"` (the CI pipeline's own install command) uses. |
| `dev` | `pytest`, `pytest-cov`, `moto[s3]` | Running the test suite. |

`server/`, `mcp-server/`, `sdk/python/`, and `sdk/typescript/` are
separate packages with their own `pyproject.toml`/`package.json` — see
each one's own README for its install/config (none of their
dependencies are added to `ragcore`'s own list; [ADR 0008](adr/0008-api-server-as-sibling-package.md)).

## Default model names

| Component | Default `model` | Override via |
|---|---|---|
| `VoyageContextEmbedder` | `voyage-context-4` | `model=` constructor arg |
| `OpenAIEmbedder` | `text-embedding-3-small` | `model=` constructor arg |
| `VoyageReranker` | `rerank-2.5` | `model=` constructor arg |
| `ClaudeGenerator` | `claude-opus-5` | `model=` constructor arg |
| Every other Claude-backed component | `ragcore.model_routing.DEFAULT_ROUTING.cheap` (`claude-haiku-4-5`) | `model=` constructor arg |

### Model routing (`ragcore/model_routing.py`)

```python
@dataclass(frozen=True)
class ModelRouting:
    cheap: str = "claude-haiku-4-5"
    strong: str = "claude-opus-5"

DEFAULT_ROUTING = ModelRouting()
```

`DEFAULT_ROUTING.strong` is what the three *synthesis-facing* Claude
components default to — `ClaudeGenerator` (the final user-facing
answer), `ClaudeVerifier` (polices what the strong model said, so
needs comparable judgment), `ClaudeCommunitySummarizer` (synthesized
prose a user may read directly). `DEFAULT_ROUTING.cheap` is what every
other Claude-backed component defaults to — extraction, rewriting,
routing/classification/judging, and intermediate (non-final-answer)
summarization: high-volume, auxiliary calls where quality matters far
less than cost/latency relative to the one call a user's actual answer
depends on. Every component still takes its own `model=...` and can be
overridden individually — this module only changes the *default* a
caller gets when it omits that argument.

## Where deeper, per-layer configuration is already documented

This file covers "how do I point `ragcore` at a model/backend/credential."
How a specific layer behaves once wired up — opt-in parameters like
`gleaning_rounds`, `enable_prompt_caching`, `context_max_tokens`,
`checkpoint_store`, `embedding_cache`, `max_hops`, `summary_caps`, and
so on — is documented where that layer itself lives, not duplicated
here:

- `CLAUDE.md` — the project's own living design-decision log; the
  single most detailed reference for *why* a given layer behaves the
  way it does.
- Each module's own docstring (`ragcore/graph/`, `ragcore/concepts/`,
  `ragcore/claims/`, `ragcore/hierarchy/`, `ragcore/contextual/`,
  `ragcore/retrieval/`, `ragcore/caching/`, `ragcore/privacy/`,
  `ragcore/injection/`, `ragcore/tenancy/`) — the opt-in parameters
  that layer's own constructor/pipeline function accepts.
- `docs/DATA_MODEL.md` — every field on every type those layers read
  and write.
- `scripts/*_demo.py` — runnable, working configuration examples for
  most of these layers (see `docs/COOKBOOK.md` once it exists, Phase 9
  item 45).
