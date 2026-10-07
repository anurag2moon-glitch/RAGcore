"""Phase 7 gap item 33: PII detection — an opt-in scan over an already-
ingested Corpus, swappable via the PiiDetector protocol. See
ragcore.privacy.pipeline.scan_corpus and ragcore.privacy.regex_detector.
RegexPiiDetector for the default, zero-new-dependency implementation.
"""

from ragcore.privacy.base import PiiDetector
from ragcore.privacy.pipeline import scan_corpus
from ragcore.privacy.regex_detector import RegexPiiDetector
from ragcore.privacy.types import PiiMatch, PiiScanReport, PiiSpan

__all__ = [
    "PiiDetector",
    "PiiMatch",
    "PiiScanReport",
    "PiiSpan",
    "RegexPiiDetector",
    "scan_corpus",
]
