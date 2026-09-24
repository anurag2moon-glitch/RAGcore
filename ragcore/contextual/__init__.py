"""Contextual chunk embeddings: before embedding, a short LLM-written blurb
situates each chunk within its document, prepended only to the text handed
to the embedder — a separate checkpointed pass, run after ingest(), the
same way ragcore.concepts/ragcore.graph attach.

    from ragcore.contextual import (
        ContextWriter, ContextStore,          # protocols
        ClaudeContextWriter, SqliteContextStore,  # default implementations
        build_contextual_embeddings, ContextReport, ContextProgress,
    )
"""

from ragcore.contextual.base import ContextStore, ContextWriter
from ragcore.contextual.claude import ClaudeContextWriter
from ragcore.contextual.pipeline import ContextProgress, ContextReport, build_contextual_embeddings
from ragcore.contextual.sqlite_store import SqliteContextStore

__all__ = [
    "ContextWriter",
    "ContextStore",
    "ClaudeContextWriter",
    "SqliteContextStore",
    "build_contextual_embeddings",
    "ContextReport",
    "ContextProgress",
]
