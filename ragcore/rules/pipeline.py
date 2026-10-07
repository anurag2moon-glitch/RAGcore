"""build_rules — Phase 10 §2's three-pass extraction pipeline: candidate
detection, structured extraction, independent verification. Mirrors
ragcore.concepts.pipeline.build_concepts / ragcore.claims.pipeline.build_claims
closely: chapter-grouped, checkpointed per (source_id, context_group_id),
LLM calls always run before any store transaction opens.

Rules whose automated verification (pass 3) comes back unsupported are
**never stored** — Phase 10's own rule, "Reject anything unsupported,"
read literally: an unsupported candidate never becomes a Rule the store
holds at all, not a Rule marked somehow "rejected." (That's a distinct,
later concept — `review_status` set by a *human* reviewer via
`ragcore.rules.review`, not yet built — every rule that does get stored
here starts life with the schema's own default, `review_status="auto"`.)

OCR-noise handling (Phase 10 §2's "Flag low-quality OCR pages instead of
extracting from them"): a chunk `ragcore.rules.ocr_quality.is_low_quality_ocr`
flags is excluded from candidate detection for that chapter, counted in
the report, and never contributes a false-negative to the chapter's
checkpoint fingerprint disappearing a real rule — the fingerprint is
computed over the *chunks actually sent* for extraction, so a later
re-run with the same (still-flagged) chunks is correctly treated as
unchanged, not endlessly retried.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

from ragcore.graph.pipeline import _group_by_context_group
from ragcore.rules.base import (
    RuleCandidateDetector,
    RuleExtractor,
    RuleStore,
    RuleVerifier,
)
from ragcore.rules.ocr_quality import is_low_quality_ocr

if TYPE_CHECKING:
    from ragcore.corpus import Corpus
    from ragcore.rules.domain import DomainPack

SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class RuleExtractionProgress:
    source_id: str
    context_group_id: str
    stage: str
    """"detecting" | "extracting" | "verifying" | "done" | "skipped" |
    "skipped_all_ocr_flagged" | "failed"."""


@dataclass(frozen=True)
class RuleExtractionReport:
    chapters_processed: int
    chapters_skipped: int
    chapters_skipped_all_ocr_flagged: int
    chunks_ocr_flagged: int
    candidates_detected: int
    rules_extracted: int
    """Candidate rules the extractor produced, before verification."""
    rules_stored: int
    """Rules that passed verification and were written to the store."""
    rules_rejected_unsupported: int
    errors: tuple[tuple[str, str, str], ...]
    total_tokens_used: int
    """Sum of `detector.usage`/`extractor.usage`/`verifier.usage`'s
    `total_tokens` (Phase 10 §8's "quotas apply to extraction jobs
    (token cost metering)") — `0` if none of the three components
    expose a `.usage` attribute (`ragcore.embedding.cost.UsageTracker`
    is an optional capability a concrete implementation attaches, same
    as every `Embedder`'s own `.usage`, never a required Protocol
    method), not an error."""


def _fingerprint(
    model: str, domain_pack: DomainPack, chunk_ids: Sequence[str]
) -> str:
    payload = "|".join([
        model, domain_pack.pack_id, domain_pack.pack_version, SCHEMA_VERSION,
        "\n".join(sorted(chunk_ids)),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_rules(
    corpus: Corpus,
    store: RuleStore,
    detector: RuleCandidateDetector,
    extractor: RuleExtractor,
    verifier: RuleVerifier,
    domain_pack: DomainPack,
    *,
    source_ids: Sequence[str] | None = None,
    resume: bool = True,
    on_progress: Callable[[RuleExtractionProgress], None] | None = None,
) -> RuleExtractionReport:
    sources = corpus.sources()
    if source_ids is not None:
        wanted = set(source_ids)
        sources = [s for s in sources if s.source_id in wanted]

    chapters_processed = 0
    chapters_skipped = 0
    chapters_skipped_all_ocr_flagged = 0
    chunks_ocr_flagged = 0
    candidates_detected = 0
    rules_extracted = 0
    rules_stored = 0
    rules_rejected_unsupported = 0
    errors: list[tuple[str, str, str]] = []

    for source in sources:
        chunks = corpus.store.get_by_source(source.source_id)
        for chapter_chunks in _group_by_context_group(chunks):
            context_group_id = chapter_chunks[0].meta.context_group_id

            clean_chunks = [c for c in chapter_chunks if not is_low_quality_ocr(c.text)]
            chunks_ocr_flagged += len(chapter_chunks) - len(clean_chunks)

            if not clean_chunks:
                chapters_skipped_all_ocr_flagged += 1
                if on_progress:
                    on_progress(
                        RuleExtractionProgress(source.source_id, context_group_id, "skipped_all_ocr_flagged")
                    )
                continue

            chunk_ids = [c.chunk_id for c in clean_chunks]
            fingerprint = _fingerprint(extractor.model, domain_pack, chunk_ids)

            if resume and store.checkpoint(source.source_id, context_group_id) == fingerprint:
                chapters_skipped += 1
                if on_progress:
                    on_progress(RuleExtractionProgress(source.source_id, context_group_id, "skipped"))
                continue

            if on_progress:
                on_progress(RuleExtractionProgress(source.source_id, context_group_id, "detecting"))

            try:
                candidates = detector.detect(clean_chunks, domain_pack=domain_pack)

                if on_progress:
                    on_progress(RuleExtractionProgress(source.source_id, context_group_id, "extracting"))
                candidate_rules = extractor.extract(candidates, domain_pack=domain_pack)

                if on_progress:
                    on_progress(RuleExtractionProgress(source.source_id, context_group_id, "verifying"))
                verifications = verifier.verify(candidate_rules, domain_pack=domain_pack)
            except Exception as e:  # noqa: BLE001 - recorded, not raised; one bad chapter shouldn't abort the corpus
                errors.append((source.source_id, context_group_id, str(e)))
                store.mark_chapter_failed(
                    source_id=source.source_id, context_group_id=context_group_id,
                    fingerprint=fingerprint, error=str(e),
                )
                if on_progress:
                    on_progress(RuleExtractionProgress(source.source_id, context_group_id, "failed"))
                continue

            candidates_detected += len(candidates)
            rules_extracted += len(candidate_rules)

            supported_by_id = {v.rule_id: v.supported for v in verifications}
            to_store = [r for r in candidate_rules if supported_by_id.get(r.rule_id, False)]
            rules_rejected_unsupported += len(candidate_rules) - len(to_store)

            store.add_rules(to_store)
            store.mark_chapter_done(
                source_id=source.source_id, context_group_id=context_group_id, fingerprint=fingerprint,
            )

            chapters_processed += 1
            rules_stored += len(to_store)
            if on_progress:
                on_progress(RuleExtractionProgress(source.source_id, context_group_id, "done"))

    total_tokens_used = sum(
        getattr(component, "usage", None).total_tokens
        for component in (detector, extractor, verifier)
        if getattr(component, "usage", None) is not None
    )

    return RuleExtractionReport(
        chapters_processed=chapters_processed,
        chapters_skipped=chapters_skipped,
        chapters_skipped_all_ocr_flagged=chapters_skipped_all_ocr_flagged,
        chunks_ocr_flagged=chunks_ocr_flagged,
        candidates_detected=candidates_detected,
        rules_extracted=rules_extracted,
        rules_stored=rules_stored,
        rules_rejected_unsupported=rules_rejected_unsupported,
        errors=tuple(errors),
        total_tokens_used=total_tokens_used,
    )
