"""Phase 7 gap item 33: prompt-injection defenses — a swappable
InjectionDetector protocol, a standalone scan_corpus() pass (same shape
as ragcore.privacy), and GuardedGenerator, a Generator decorator that
screens retrieved context before generation. See ragcore.injection.guard
and ragcore.injection.regex_detector.RegexInjectionDetector for the
default, zero-new-dependency implementation.
"""

from ragcore.injection.base import InjectionDetector
from ragcore.injection.guard import GuardedGenerator
from ragcore.injection.pipeline import scan_corpus
from ragcore.injection.regex_detector import RegexInjectionDetector
from ragcore.injection.types import InjectionMatch, InjectionScanReport, InjectionSpan

__all__ = [
    "GuardedGenerator",
    "InjectionDetector",
    "InjectionMatch",
    "InjectionScanReport",
    "InjectionSpan",
    "RegexInjectionDetector",
    "scan_corpus",
]
