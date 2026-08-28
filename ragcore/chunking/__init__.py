"""Chunker protocol and the default StructuralChunker implementation."""

from ragcore.chunking.base import Chunker
from ragcore.chunking.structural import StructuralChunker

__all__ = ["Chunker", "StructuralChunker"]
