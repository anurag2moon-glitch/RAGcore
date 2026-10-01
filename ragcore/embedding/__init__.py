"""Embedder protocol and backend implementations (Voyage, OpenAI, and
later a self-hosted adapter)."""

from ragcore.embedding.base import Embedder
from ragcore.embedding.openai import OpenAIEmbedder
from ragcore.embedding.voyage import VoyageContextEmbedder

__all__ = ["Embedder", "OpenAIEmbedder", "VoyageContextEmbedder"]
