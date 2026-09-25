# ragcore

Reusable retrieval layer for RAG: structure-aware ingestion for long-form
books (PDF, EPUB, DOCX, TXT, MD), a swappable embedder and vector store, an
entity/relation GraphRAG layer, a concept/theme layer with community
detection and summarization, hybrid (BM25 + vector) retrieval, contextual
chunk embeddings, multi-hop iterative retrieval, cross-encoder reranking,
chunk→section→chapter→book hierarchical summarization, MMR diversity
reranking, citation verification, a general-purpose evaluation harness, and
incremental ingestion with basic versioning/rollback.

```python
from ragcore import Corpus

corpus = Corpus.open("my-library")
corpus.ingest("book.epub")
results = corpus.retrieve("what does he say about attention", k=10)
for r in results:
    print(r.chunk.meta.section_path, r.score)
```

Every layer attaches in front of retrieval/generation without changing
`Corpus` itself — see `ragcore/graph`, `ragcore/concepts`, `ragcore/lexical`,
`ragcore/contextual`, `ragcore/hierarchy`, `ragcore/reranking`,
`ragcore/eval`, `ragcore/versioning`, `ragcore/verification`,
`ragcore/claims` (claim extraction and cross-source linking — storage only,
no retriever of its own), `ragcore/conversation` (session transcripts and
context-aware query rewriting on top of an existing Corpus), and
`ragcore/retrieval/{graph,concept,hybrid,iterative,cross_encoder,hierarchy,mmr,stack}.py`
(`ragcore.retrieval.stack.default_stack()` wires the full default retriever
stack in one call). Run `python scripts/graph_demo.py`,
`python scripts/concepts_demo.py`, or `python scripts/run_eval.py --demo`
for a worked example.

See the design doc (module structure, abstraction boundaries, metadata
schema, and how each layer attaches) for the full architecture.
