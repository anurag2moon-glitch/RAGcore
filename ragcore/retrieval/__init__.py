"""Retriever protocol and the default VectorRetriever implementation."""

from ragcore.retrieval.base import Retriever
from ragcore.retrieval.vector import VectorRetriever

__all__ = ["Retriever", "VectorRetriever"]
