"""Lexical (keyword/BM25) retrieval channel, attached alongside the vector
store — see ragcore.retrieval.hybrid.HybridRetriever for how the two fuse.

    from ragcore.lexical import LexicalStore, LexicalHit, Fts5LexicalStore, build_lexical_index
"""

from ragcore.lexical.base import LexicalStore
from ragcore.lexical.fts5 import Fts5LexicalStore
from ragcore.lexical.pipeline import LexicalIndexReport, build_lexical_index
from ragcore.lexical.types import LexicalHit

__all__ = [
    "LexicalHit",
    "LexicalStore",
    "Fts5LexicalStore",
    "build_lexical_index",
    "LexicalIndexReport",
]
