# Data model reference

Phase 9 deliverable. Every dataclass a consumer of `ragcore` is likely
to construct, read, or persist, grouped by module. Field types and
defaults are copied directly from the source — if this drifts from
`ragcore/*/types.py`, the source is authoritative; update this file
alongside it. Non-obvious fields link to the ADR that explains the
reasoning rather than repeating it here.

All types below are `@dataclass(frozen=True)` unless noted.

## Core pipeline (`ragcore/types.py`)

### `Block`
One extracted unit of text with its structural position, as a `Loader`
produces it.

| Field | Type | Notes |
|---|---|---|
| `text` | `str` | |
| `page` | `int \| None` | Set only by loaders with a real page concept (`PdfLoader`). |
| `section_path` | `tuple[str, ...]` | Full breadcrumb to this block's position — blocks are flat, not a tree. |
| `page_hash` | `str \| None = None` | `sha256(text)`, set only alongside `page`. Powers page-level dedupe. |

### `Document`
One fully-loaded file.

| Field | Type | Notes |
|---|---|---|
| `source_id` | `str` | Content-addressed (`sha256` of file bytes) — see [ADR 0010](adr/0010-content-addressed-identity.md). |
| `path` | `str` | |
| `title` | `str \| None` | |
| `source_type` | `SourceType` | `Literal["pdf","epub","docx","txt","md","html"]` — closed, every `Loader` emits exactly one of these six. |
| `blocks` | `list[Block]` | |
| `meta` | `dict[str, Any] = {}` | Loader-specific extras. |

### `ChunkMeta`
Metadata carried by every chunk from ingestion through retrieval. The
largest type in this codebase — grouped below by the comment headers
in the source itself.

| Field | Type | Notes |
|---|---|---|
| **provenance** | | |
| `source_id` | `str` | |
| `source_path` | `str` | |
| `source_title` | `str \| None` | |
| `source_type` | `ChunkOrigin` | `SourceType \| Literal["synthetic"]` — `"synthetic"` marks a chunk `ragcore` generated itself (a community/hierarchy summary), never a real `Loader` output. |
| **book structure** | | |
| `section_path` | `tuple[str, ...]` | |
| `chapter_title` | `str \| None` | Display-only, can be noisy (see `CLAUDE.md`'s "chapter_title vs chapter_index"). Never filter on this. |
| `chapter_index` | `int \| None` | Stable ordinal — filter/range-query on this instead. |
| `section` | `str \| None` | |
| `page_start` / `page_end` | `int \| None` | |
| **position & identity** | | |
| `chunk_index` | `int` | |
| `context_group_id` | `str` | The contextualization unit a `Chunker` grouped this chunk into. |
| `char_start` / `char_end` | `int` | |
| `token_count` | `int` | |
| **multi-tenancy (Phase 7)** | | |
| `tenant_id` | `str \| None = None` | Opt-in, additive. Mechanism, not an enforced guarantee by itself — see [ADR 0006](adr/0006-tenant-id-foundation-not-enforcement.md) and `ragcore.tenancy.TenantScopedCorpus`. |
| **licensing (Phase 7)** | | |
| `license_id` | `str \| None = None` | Open vocabulary (e.g. `"CC-BY-4.0"`). Named `license_id`, not `license`, to avoid shadowing the Python builtin. Same "mechanism, not enforcement" positioning as `tenant_id`. |
| **extensibility** | | |
| `extra` | `dict[str, Any] = {}` | Add-on layers stamp provenance here (e.g. `extra["ragcore_kind"] == "community_summary"`, `extra["context_blurb"]`, `extra["hop_index"]`, `extra["rrf_score"]`, `extra["injection_flagged"]`) rather than growing new first-class fields for every optional layer. |

### `Chunk`
| Field | Type | Notes |
|---|---|---|
| `chunk_id` | `str` | `f"{source_id}:{chunk_index}"` — positional within a content-addressed source, **not** a hash of the chunk's own text; see [ADR 0010](adr/0010-content-addressed-identity.md)'s documented gap. |
| `text` | `str` | |
| `meta` | `ChunkMeta` | |
| `vector` | `Vector \| None = None` | `None` as produced by a `Chunker`; always populated once an `Embedder`/`VectorStore` has touched it. |

### `ChunkGroup`
| Field | Type | Notes |
|---|---|---|
| `group_id` | `str` | |
| `chunks` | `list[Chunk]` | The unit one `Embedder.embed_groups()` call contextualizes jointly. |

### `Retrieved`
| Field | Type | Notes |
|---|---|---|
| `chunk` | `Chunk` | |
| `score` | `float` | `1/(1+distance)` cosine-derived scale, consistently, across every retrieval layer — see `ragcore.retrieval.scoring`. |
| `graph` | `GraphContext \| None = None` | Populated only when a `GraphRetriever` is wired on top. |

### `SourceInfo`, `IngestProgress`, `IngestReport`
Reporting types returned by `Corpus.sources()`/`Corpus.ingest()`.
Fields are self-explanatory from their names; see the server's
`IngestResult`/`SourceInfo` Pydantic models (`server/app/models.py`)
for the HTTP JSON projection of these two.

## GraphRAG (`ragcore/graph/types.py`)

### `Entity`
| Field | Type | Notes |
|---|---|---|
| `entity_id` | `str` | Content-addressed: `f"entity:{type}:{normalize(name)}"` — see [ADR 0010](adr/0010-content-addressed-identity.md). |
| `name` | `str` | |
| `entity_type` | `str` | Open vocabulary. |
| `source_chunk_ids` | `tuple[str, ...]` | Accumulated across merges — see [ADR 0011](adr/0011-graph-merge-not-replace.md). |
| `extra` | `dict[str, Any] = {}` | Carries `extra["aliases"]` and, since the relational-entity mitigation, `extra["relations"]`. |

### `Relation`
| Field | Type | Notes |
|---|---|---|
| `relation_id` | `str` | Content-addressed from the two endpoint entity ids + type. |
| `source_entity_id` / `target_entity_id` | `str` | |
| `relation_type` | `str` | |
| `source_chunk_ids` | `tuple[str, ...]` | |
| `extra` | `dict[str, Any] = {}` | |

### `GraphContext`
| Field | Type | Notes |
|---|---|---|
| `entities` | `tuple[str, ...]` | Which entities tied this chunk to the query. |
| `path` | `tuple[str, ...]` | The traversal path that reached it. |

## Concepts (`ragcore/concepts/types.py`)

### `Concept`
| Field | Type | Notes |
|---|---|---|
| `concept_id` | `str` | `f"concept:{normalize(label)}"`. |
| `label` | `str` | A proposition (e.g. "consequences of unchecked ambition"), never a bare keyword. |
| `description` | `str` | What's actually embedded for resolution/mapping. |
| `source_chunk_ids` | `tuple[str, ...]` | |
| `extra` | `dict[str, Any] = {}` | |
| `vector` | `tuple[float, ...] \| None = None` | Set only when genuinely new (not a content-address match, not a resolver merge) — never clobbers an existing stored vector with `None`. |

### `ConceptEdge`
| Field | Type | Notes |
|---|---|---|
| `edge_id` | `str` | `f"edge:{a}:{b}"`, sorted — one canonical id per unordered pair. |
| `source_concept_id` / `target_concept_id` | `str` | Sorted; "source"/"target" carry no direction here. |
| `weight` | `int` | **Derived** as `COUNT(DISTINCT chunk_id)` on read, never an incrementing counter (would double-count a re-run). |
| `source_chunk_ids` | `tuple[str, ...]` | |
| `extra` | `dict[str, Any] = {}` | |

### `CommunitySummary`
| Field | Type | Notes |
|---|---|---|
| `community_id` | `str` | Content-addressed over sorted member concept ids. |
| `label` / `summary` | `str` | |
| `agreements` / `divergences` | `tuple[str, ...]` | |
| `concept_ids` / `source_chunk_ids` / `source_ids` | `tuple[str, ...]` | `source_ids` are distinct book/document sources sampled, round-robin, so one long book can't dominate. |
| `model` / `prompt_version` / `created_at` | `str` | |
| `vector` | `tuple[float, ...] \| None = None` | |
| `chunk_fingerprint` | `str \| None = None` | `sha256` of every reachable chunk_id at last computation — catches a source re-ingest that leaves community membership unchanged but underlying text stale. |

## Claims (`ragcore/claims/types.py`)

### `SourceClaim`
| Field | Type | Notes |
|---|---|---|
| `claim_id` | `str` | `f"claim:{sha256(source_id\|chunk_id\|normalized_text)[:16]}"` — scoped by chunk, unlike entity/concept ids, so two independent sources asserting the same thing stay two rows (the agreement signal this layer exists to capture). |
| `concept_ids` | `tuple[str, ...]` | Anchors into the concept layer — `build_claims()` must run after `build_concepts()`. |
| `source_id` / `chunk_id` | `str` | |
| `text` | `str` | A verbatim span, never a paraphrase. |
| `author_confidence` | `ClaimConfidence` | `Literal["firm","hedged"]`, judged only from that exact passage's own wording. |
| `model` / `prompt_version` / `created_at` | `str` | |
| `vector` | `tuple[float, ...] \| None = None` | |
| `tenant_id` | `str \| None = None` | Taken directly from the source chunk — Phase 7 Tier 1 (see [ADR 0007](adr/0007-tenant-id-on-remaining-stores-two-tiers.md)). |
| `extra` | `dict[str, Any] = {}` | Not currently persisted by `SqliteClaimStore` — a pre-existing gap, not something Phase 7 introduced. |

### `ClaimRelation`
| Field | Type | Notes |
|---|---|---|
| `relation_id` | `str` | Type baked into the id: `agrees`/`conflicts` sort their endpoints (`f"claimrel:{type}:{a}:{b}"`, symmetric); `extends` keeps direction (`f"claimrel:extends:{source}:{target}"`). |
| `source_claim_id` / `target_claim_id` | `str` | |
| `relation_type` | `ClaimRelationType` | `Literal["agrees","conflicts","extends"]`. |
| `rationale` | `str` | Required, never folded into the id. |
| `model` / `prompt_version` / `created_at` | `str` | |

## Hierarchy (`ragcore/hierarchy/types.py`)

### `HierarchyNode`
| Field | Type | Notes |
|---|---|---|
| `node_id` | `str` | **Positional**, not content-addressed — `f"hierarchy:{level}:{source_id}:..."`. A node's identity is inherently tied to one position in one book's own structure. |
| `level` | `HierarchyLevel` | `Literal["section","chapter","book"]`. |
| `source_id` | `str` | |
| `label` / `summary` | `str` | |
| `child_ids` | `tuple[str, ...]` | Chunk ids (section level) or child node ids (chapter/book level). |
| `source_chunk_ids` | `tuple[str, ...]` | Every leaf chunk under this node, regardless of level. |
| `model` / `prompt_version` / `created_at` | `str` | |
| `vector` | `tuple[float, ...] \| None = None` | Filled in by `build_hierarchy()` after the summarizer returns, same discipline as `Concept.vector`. |
| `tenant_id` | `str \| None = None` | Filled in by `build_hierarchy()` from the owning source's chunks — Phase 7 Tier 1. |

## Eval harness (`ragcore/eval/types.py`)

`GoldQuestion` / `GoldPassage` — a storable gold-question format;
`GoldPassage` matches a retrieved chunk at one of three granularities
(exact `chunk_ids`, then `(source_id, chapter_index)`, then
`source_path` alone) so a re-chunked corpus doesn't silently zero out
an old gold set. `RetrievalMetrics` (`recall_at_k`/`precision_at_k`/
`ndcg_at_k`/`mrr`/`hit_rate_at_k`, plus Phase 6's
`latency_p50/95/99_ms` and `total_tokens_used: int | None`) and
`GroundednessMetrics` aggregate across a `QuestionResult` per question
(adds `latency_ms`) into one `EvalReport`.

## Phase 7 add-ons

| Module | Type | Key fields |
|---|---|---|
| `ragcore.privacy.types` | `PiiSpan` | `pii_type: str`, `start: int`, `end: int` — chunk-agnostic, as a `PiiDetector` reports it. |
| | `PiiMatch` | Adds `chunk_id`/`source_id`. Deliberately **no copy of the matched text** — see the type's own docstring for why a scan report must never become a second place PII ends up persisted in plaintext. |
| | `PiiScanReport` | `chunks_scanned: int`, `matches: tuple[PiiMatch, ...]`, `counts_by_type()`. |
| `ragcore.injection.types` | `InjectionSpan` / `InjectionMatch` / `InjectionScanReport` | Same shape as the PII types above, over `pattern: str` instead of `pii_type`. |
| `ragcore.tenancy.types` | `TenantDeletionReport` | `tenant_id: str`, `source_ids: tuple[str, ...]`, `chunks_deleted: int` — returned by `TenantScopedCorpus.hard_delete()`. |

## Server / SDK JSON projections

`server/app/models.py` and both SDKs' `types.py`/`types.ts` are
**deliberate projections**, not 1:1 mirrors, of the above — see
`server/app/models.py`'s own module docstring. `RetrievedChunkResponse`
exposes 8 of `ChunkMeta`'s ~20 fields; extend it as a real client's
needs become concrete, not speculatively ahead of that.
