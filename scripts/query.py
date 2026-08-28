"""Ad-hoc CLI for querying an existing ragcore corpus. Not part of the
library's public API — just a convenience for manually checking retrieval
(and, with --ask, generation) quality against a corpus you've already
ingested.

Usage:
    VOYAGE_API_KEY=... python scripts/query.py samples/library.lancedb "your query" [k]
    VOYAGE_API_KEY=... ANTHROPIC_API_KEY=... python scripts/query.py samples/library.lancedb "your query" [k] --ask
"""

from __future__ import annotations

import sys

from ragcore.corpus import Corpus
from ragcore.embedding.voyage import VoyageContextEmbedder
from ragcore.stores.lancedb import LanceDBStore


def main() -> None:
    args = list(sys.argv[1:])
    ask = "--ask" in args
    if ask:
        args.remove("--ask")

    if len(args) < 2:
        print(f"usage: {sys.argv[0]} <corpus-path> <query> [k] [--ask]")
        raise SystemExit(1)

    db_path, query = args[0], args[1]
    k = int(args[2]) if len(args) > 2 else 5

    embedder = VoyageContextEmbedder(output_dimension=1024, min_request_interval=21.0)
    store = LanceDBStore(db_path, dim=embedder.dim)

    generator = None
    if ask:
        from ragcore.generation.claude import ClaudeGenerator

        generator = ClaudeGenerator()

    corpus = Corpus(store, embedder, generator=generator)

    print(f"corpus has {corpus.count()} chunks across {len(corpus.sources())} source(s)\n")

    if ask:
        answer = corpus.ask(query, k=k)
        print(answer.text)
        print("\n--- grounded in ---")
        for r in answer.citations:
            m = r.chunk.meta
            print(f"  score={r.score:.4f}  chapter_index={m.chapter_index}  chapter_title={m.chapter_title!r}  page={m.page_start}")
        return

    for r in corpus.retrieve(query, k=k):
        m = r.chunk.meta
        print(f"score={r.score:.4f}  chapter_index={m.chapter_index}  chapter_title={m.chapter_title!r}  page={m.page_start}")
        print(f"{r.chunk.text}\n")
        print("-" * 70)


if __name__ == "__main__":
    main()
