# System diagram

Phase 9 deliverable. Renders natively on GitHub; paste into
[mermaid.live](https://mermaid.live) for a standalone view. Reflects
the codebase as of commit `e8e8909` — update this alongside
`docs/ARCHITECTURE.md` when the shape of a pipeline changes, not just
when a module is added.

## Core pipeline + add-on layers

A consuming project builds exactly one `Corpus` and drives everything
through it. The five add-on layers (graph, concepts, claims, hierarchy,
contextual) are separate, optional `build_*()` passes that run *after*
`ingest()`, never inside it — see `CLAUDE.md`'s "Concept layer" section
for why. `Corpus.retriever` is a swappable attribute: any subset of the
decorator chain on the right can be wired on top of a plain
`VectorRetriever`/`HybridRetriever`, in the order
`ragcore.retrieval.stack.default_stack()` documents.

```mermaid
flowchart TD
    subgraph ingest["Ingestion (Corpus.ingest)"]
        Loader["Loader\n(pdf/epub/docx/txt/md/html)"] --> Chunker["Chunker\n(Structural / Semantic)"]
        Chunker --> Embedder["Embedder\n(Voyage / OpenAI)"]
        Embedder --> VectorStore["VectorStore\n(LanceDB)"]
    end

    Corpus(["Corpus"]) -->|".ingest()"| ingest
    Corpus -->|".retrieve() / .ask()"| Retriever

    subgraph retrieve["Retrieval decorator chain (Corpus.retriever)"]
        direction TB
        VR["VectorRetriever\nor HybridRetriever\n(+ Fts5LexicalStore, RRF)"]
        VR --> GraphR["GraphRetriever\n(opt-in)"]
        GraphR --> ConceptR["ConceptRetriever\n(opt-in)"]
        ConceptR --> HierR["HierarchyRetriever\n(opt-in)"]
        HierR --> IterR["IterativeRetriever\n(opt-in, multi-hop)"]
        IterR --> XEnc["CrossEncoderReranker\n(opt-in)"]
        XEnc --> MMR["MMRReranker"]
        MMR --> Cache["CachingRetriever\n(opt-in, wraps the whole chain)"]
    end
    Retriever(["Retriever"]) --> VR

    Corpus -->|".ask()"| Generator["Generator\n(ClaudeGenerator)"]
    Generator -.->|"opt-in decorator"| Verify["VerifyingGenerator"]
    Generator -.->|"opt-in decorator"| Guard["GuardedGenerator\n(ragcore.injection)"]

    subgraph addon["Optional add-on passes (run after ingest(), not inside it)"]
        direction LR
        BuildGraph["build_graph()"] --> GraphStore["SqliteGraphStore /\nKuzuGraphStore"]
        BuildConcepts["build_concepts()"] --> ConceptStore["SqliteConceptStore"]
        BuildClaims["build_claims()"] --> ClaimStore["SqliteClaimStore"]
        BuildHier["build_hierarchy()"] --> HierStore["SqliteHierarchyStore"]
        BuildCtx["build_contextual_embeddings()"] --> CtxStore["SqliteContextStore"]
    end
    VectorStore -.->|"chunks feed every add-on extractor"| addon
    GraphStore --> GraphR
    ConceptStore --> ConceptR
    HierStore --> HierR

    ScanPII["scan_corpus()\n(ragcore.privacy)"] -.->|"reads chunks, writes nothing back"| VectorStore
    ScanInj["scan_corpus()\n(ragcore.injection)"] -.->|"reads chunks, writes nothing back"| VectorStore
```

## Multi-tenancy + the sibling packages (Phase 7/8)

`ragcore/` itself never grows a server, a CLI, or global state
(`docs/adr/0009-library-not-application.md`). Everything below the
dashed line is a *separate* package that depends on `ragcore`, not code
inside it.

```mermaid
flowchart TD
    RawCorpus(["Corpus"]) --> Scoped["TenantScopedCorpus\n(ragcore.tenancy)\nforces tenant_id on every call"]
    Scoped -->|".hard_delete()"| AllStores["vector store +\noptional graph/concept/\nclaim/context/hierarchy/\nlexical stores"]

    RawCorpus -.-|"zero new deps added to ragcore's\nown pyproject.toml for any of these"| boundary["separate sibling packages\n(docs/adr/0008)"]

    boundary --> Server["server/\nFastAPI reference server\n(single-tenant, no auth yet)"]
    Server -->|"auto-generated"| OpenAPI["/docs\nOpenAPI spec"]
    Server --> PySdk["sdk/python/\nragcore_client.RagcoreClient"]
    Server --> TsSdk["sdk/typescript/\nRagcoreClient"]
    boundary --> McpServer["mcp-server/\n4 tools: ingest_file,\nretrieve, ask, list_sources"]
```

## What's deliberately not shown

- The exact id-generation formulas for `chunk_id`/`source_id`/
  `entity_id`/`concept_id`/`claim_id` — see
  `docs/adr/0010-content-addressed-identity.md`.
- Checkpoint/resume plumbing for every `build_*()` pass — each one's
  own module docstring covers its exact fingerprint scheme.
- The full eval harness (`ragcore.eval`) and versioning/snapshot
  orchestration (`ragcore.versioning`) — see `docs/RUNBOOK.md` for the
  latter.
