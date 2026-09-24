"""Reranker protocol + default implementation, for
ragcore.retrieval.cross_encoder.CrossEncoderReranker.

    from ragcore.reranking import Reranker, RerankHit, VoyageReranker
"""

from ragcore.reranking.base import Reranker, RerankHit
from ragcore.reranking.voyage import VoyageReranker

__all__ = ["Reranker", "RerankHit", "VoyageReranker"]
