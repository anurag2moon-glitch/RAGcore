"""Post-generation citation/groundedness verification, attached to a
Generator the same way ragcore.retrieval decorators attach to a Retriever.

    from ragcore.verification import (
        ClaimCheck, Verification,          # types
        Verifier,                          # protocol
        ClaudeVerifier, VerifyingGenerator, # default implementation + the Generator decorator
    )

    corpus.generator = VerifyingGenerator(corpus.generator, ClaudeVerifier())
"""

from ragcore.verification.base import Verifier
from ragcore.verification.claude import ClaudeVerifier, VerifyingGenerator
from ragcore.verification.types import ClaimCheck, ClaimStatus, Verification

__all__ = [
    "ClaimCheck",
    "ClaimStatus",
    "Verification",
    "Verifier",
    "ClaudeVerifier",
    "VerifyingGenerator",
]
