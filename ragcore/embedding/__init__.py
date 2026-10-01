"""Embedder protocol and backend implementations (Voyage, OpenAI, and
later a self-hosted adapter), plus the cost (ragcore.embedding.cost) and
content-hash cache (ragcore.embedding.cache) building blocks a Corpus can
opt into."""

from ragcore.embedding.base import Embedder
from ragcore.embedding.cache import EmbeddingCache, SqliteEmbeddingCache
from ragcore.embedding.cost import UsageTracker
from ragcore.embedding.openai import OpenAIEmbedder
from ragcore.embedding.voyage import VoyageContextEmbedder

__all__ = [
    "Embedder",
    "EmbeddingCache",
    "OpenAIEmbedder",
    "SqliteEmbeddingCache",
    "UsageTracker",
    "VoyageContextEmbedder",
]
