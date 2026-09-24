"""link_claims — corpus-global pass finding agree/conflict/extends relations
between SourceClaims anchored to the same concept.

Deliberately a separate file from ragcore.claims.claude (the extractor and
linker implementations) and from any "detection" step — unlike
ragcore.concepts.communities, which pairs Louvain detection with
summarization in one file, linking has no clustering step of its own to
justify that shape; every claim's concept_ids already say which group it's
in.

Bounding strategy: a popular concept can accumulate hundreds of claims
across a large corpus, so this does NOT send a concept's full claim set to
the linker in one call. ragcore.concepts.communities._sample_chunks_round_robin
was considered and rejected for this — lossy random sampling is fine for a
synthesized summary, but here it would make an unchanged corpus produce a
DIFFERENT relation set on every run, which is nondeterminism this codebase
otherwise carefully avoids. Instead this reuses the embedding-shortlist
pattern ragcore.concepts.mapper.ClaudeConceptMapper already uses over
EmbeddingConceptMapper: each claim's own persisted vector picks its
top-`shortlist_k` nearest neighbors within the concept group as candidate
partners, and those candidate pairs are batched into groups of `batch_size`
claims per linker.link() call. Deterministic, sub-quadratic, no new
dependency. Duplicate relations produced by overlapping batches are
harmless — ClaimStore.put_relations merges by content-addressed
relation_id.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from ragcore.claims.base import ClaimLinker, ClaimStore
from ragcore.claims.types import SourceClaim
from ragcore.graph.resolver import _cosine

PROMPT_VERSION = "claim-links-v1"
SCHEMA_VERSION = "1"

DEFAULT_SHORTLIST_K = 10
DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_CLAIMS_PER_CONCEPT = 200
"""A coverage bound, not a quality knob: a concept with more than this many
claims will have some claims never considered for linking at all, rather
than being linked "less accurately." Raise it if a persona's corpus
genuinely needs deeper coverage on a single concept."""


@dataclass(frozen=True)
class LinkReport:
    concepts_processed: int
    concepts_skipped: int
    relations_found: int


def _fingerprint(
    model: str, claim_ids: Sequence[str], shortlist_k: int, batch_size: int
) -> str:
    payload = "|".join([
        model, PROMPT_VERSION, SCHEMA_VERSION, f"shortlist_k={shortlist_k}",
        f"batch_size={batch_size}", "\n".join(sorted(claim_ids)),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate_batches(
    claims: Sequence[SourceClaim], *, shortlist_k: int, batch_size: int
) -> list[list[SourceClaim]]:
    """Groups claims into overlapping batches such that every claim is
    batched together with its own nearest neighbors at least once. Claims
    with no vector (shouldn't happen in practice — build_claims always
    embeds new claims — but handled defensively) are dropped from
    consideration rather than crashing the whole concept's linking run."""
    vectored = [c for c in claims if c.vector is not None]
    if len(vectored) < 2:
        return []

    # For each claim, its own shortlist of nearest neighbors (by index into
    # `vectored`), sorted by descending cosine similarity.
    neighbor_lists: list[list[int]] = []
    for i, claim in enumerate(vectored):
        scored = [
            (j, _cosine(claim.vector, other.vector))
            for j, other in enumerate(vectored)
            if j != i
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        neighbor_lists.append([j for j, _score in scored[:shortlist_k]])

    # Union each claim with its shortlist into one candidate set, then chunk
    # that set into fixed-size batches — every claim ends up batched with at
    # least its own nearest neighbors, though not necessarily all in the
    # same single batch once shortlists overlap widely.
    candidate_indices: set[int] = set()
    for i, neighbors in enumerate(neighbor_lists):
        candidate_indices.add(i)
        candidate_indices.update(neighbors)
    ordered = sorted(candidate_indices)

    batches: list[list[SourceClaim]] = []
    for start in range(0, len(ordered), batch_size):
        batch_indices = ordered[start : start + batch_size]
        batches.append([vectored[i] for i in batch_indices])

    # Also ensure every claim and its own shortlist co-occur in at least one
    # batch, even if the fixed-size chunking above split them apart — append
    # a small extra batch per claim whose neighbors landed elsewhere. This
    # keeps the guarantee ("every claim is compared against its nearest
    # neighbors") without requiring the fixed-size chunks above to be
    # neighbor-aware, which would make batch composition much harder to
    # reason about.
    index_to_batch = {}
    for batch_i, batch in enumerate(batches):
        for claim in batch:
            index_to_batch[claim.claim_id] = batch_i
    for i, neighbors in enumerate(neighbor_lists):
        claim = vectored[i]
        own_batch = index_to_batch[claim.claim_id]
        missing = [
            vectored[j] for j in neighbors if index_to_batch.get(vectored[j].claim_id) != own_batch
        ]
        if missing:
            batches.append([claim, *missing][: batch_size + 1])

    return batches


def link_claims(
    store: ClaimStore,
    linker: ClaimLinker,
    *,
    concept_ids: Sequence[str] | None = None,
    resume: bool = True,
    shortlist_k: int = DEFAULT_SHORTLIST_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_claims_per_concept: int = DEFAULT_MAX_CLAIMS_PER_CONCEPT,
) -> LinkReport:
    wanted = store.concept_ids_with_claims(min_count=2)
    if concept_ids is not None:
        allowed = set(concept_ids)
        wanted = [cid for cid in wanted if cid in allowed]

    concepts_processed = 0
    concepts_skipped = 0
    relations_found = 0

    for concept_id in wanted:
        claims = store.claims_for_concept(concept_id)[:max_claims_per_concept]
        claim_ids = [c.claim_id for c in claims]
        fingerprint = _fingerprint(linker.model, claim_ids, shortlist_k, batch_size)

        if resume and store.link_checkpoint(concept_id) == fingerprint:
            concepts_skipped += 1
            continue

        batches = _candidate_batches(claims, shortlist_k=shortlist_k, batch_size=batch_size)
        for batch in batches:
            relations = linker.link(batch)
            if relations:
                store.put_relations(relations)
                relations_found += len(relations)

        store.mark_concept_linked(concept_id=concept_id, fingerprint=fingerprint)
        concepts_processed += 1

    return LinkReport(
        concepts_processed=concepts_processed,
        concepts_skipped=concepts_skipped,
        relations_found=relations_found,
    )
