"""Shared token counting for the chunker and, where a backend has no
cheaper native counting endpoint, the embedder's batching logic.

Uses tiktoken's cl100k_base encoding as an approximation. This is not the
exact tokenizer voyage-context-4 (or any other backend) uses internally,
but token *budgets* only need to be a stable, reasonably-close proxy —
chunk boundaries don't need to land on the embedder's exact token count,
they need to stay safely under its limits. cl100k_base is a widely-used,
dependency-light approximation that satisfies that.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoding().encode(text))
