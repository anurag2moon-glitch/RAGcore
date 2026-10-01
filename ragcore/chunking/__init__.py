"""Chunker protocol and its two implementations: StructuralChunker
(default, token-budget packing) and SemanticChunker (embedding-boundary
breakpoints — see ragcore/chunking/semantic.py)."""

from ragcore.chunking.base import Chunker
from ragcore.chunking.semantic import SemanticChunker
from ragcore.chunking.structural import StructuralChunker

__all__ = ["Chunker", "SemanticChunker", "StructuralChunker"]
