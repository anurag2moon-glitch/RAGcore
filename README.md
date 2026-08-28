# ragcore

Reusable retrieval layer for RAG: structure-aware ingestion for long-form
books (PDF, EPUB, DOCX, TXT, MD), a swappable embedder and vector store, and
a GraphRAG layer designed in but not yet built.

```python
from ragcore import Corpus

corpus = Corpus.open("my-library")
corpus.ingest("book.epub")
results = corpus.retrieve("what does he say about attention", k=10)
for r in results:
    print(r.chunk.meta.section_path, r.score)
```

See the design doc (module structure, abstraction boundaries, metadata
schema, and how the graph layer attaches later) for the full architecture.
