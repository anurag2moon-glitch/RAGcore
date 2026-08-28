"""Embedder protocol and backend implementations (Voyage, and later a
self-hosted adapter)."""

from ragcore.embedding.base import Embedder
from ragcore.embedding.voyage import VoyageContextEmbedder

__all__ = ["Embedder", "VoyageContextEmbedder"]
