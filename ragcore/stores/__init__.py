"""VectorStore protocol and backend implementations (LanceDB, and later
others)."""

from ragcore.stores.base import VectorStore
from ragcore.stores.lancedb import LanceDBStore

__all__ = ["VectorStore", "LanceDBStore"]
