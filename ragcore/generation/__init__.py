"""Generator protocol and backend implementations (Claude)."""

from ragcore.generation.base import Answer, Generator
from ragcore.generation.claude import ClaudeGenerator

__all__ = ["Answer", "Generator", "ClaudeGenerator"]
