"""Data types for the concept/theme layer.

Shaped after ragcore.graph.types (Entity/Relation): Concept and ConceptEdge
are the concept-graph analogues of Entity and Relation, right down to the
content-addressed id scheme and the merge-not-replace storage contract (see
ragcore.concepts.sqlite_store). CommunitySummary is the one new shape this
layer adds — a synthesized cross-source summary, not something the entity
layer has an equivalent of.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Concept:
    """An abstract idea or theme discussed across one or more chunks — not a
    keyword, a proposition (e.g. "consequences of unchecked ambition", never
    just "ambition"). See ragcore.concepts.claude_extractor for the
    extraction prompt that enforces this distinction."""

    concept_id: str
    """Content-addressed: f"concept:{_normalize(label)}" — see
    ragcore.graph.claude_extractor._normalize, reused here so an exact
    repeated label collapses for free, the same way entity names do."""

    label: str
    """The abstract proposition itself, in a few words."""

    description: str
    """One sentence of disambiguating context. Also the text actually
    embedded for resolution (ragcore.concepts.resolver) and query-time
    mapping (ragcore.concepts.mapper) — see ConceptStore.concept_vector."""

    source_chunk_ids: tuple[str, ...]
    """Chunks in which this concept was discussed, as extracted in one
    build_concepts() call. A ConceptStore accumulates these across calls via
    concept_chunks (same "merge, not replace" contract as
    ragcore.graph.sqlite_store.SqliteGraphStore.add_entities)."""

    extra: dict[str, Any] = field(default_factory=dict)

    vector: tuple[float, ...] | None = None
    """Set only for a concept ragcore.concepts.pipeline.build_concepts just
    determined is genuinely new (not an exact content-addressed match and
    not a resolver merge into an existing concept) — computed once there and
    persisted by ConceptStore.add_chapter_concepts. None on every other
    Concept, including one about to be merged into an existing id that
    already has a stored vector; the store never clobbers a stored vector
    with None."""


@dataclass(frozen=True)
class ConceptEdge:
    """An undirected co-occurrence edge between two concepts: both were
    discussed in the same chunk. Derived by the extractor from co-occurrence
    within each chunk — never asked of the model directly (see
    ragcore.concepts.claude_extractor)."""

    edge_id: str
    """Content-addressed over the two (sorted) concept ids: an undirected
    edge has exactly one canonical id regardless of extraction order —
    f"edge:{a}:{b}" with a < b."""

    source_concept_id: str
    target_concept_id: str
    """Sorted so source_concept_id < target_concept_id — "source"/"target"
    here means nothing directional, just a stable pair ordering."""

    weight: int
    """Number of distinct chunks in which both concepts co-occur. A
    ConceptStore accumulates this across calls rather than overwriting it."""

    source_chunk_ids: tuple[str, ...]
    """Chunks contributing to this edge's weight, as extracted in one
    build_concepts() call."""

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommunitySummary:
    """A precomputed, LLM-synthesized summary of one community (cluster) of
    related concepts — what they collectively say, where sources agree,
    where they diverge. See ragcore.concepts.communities.build_community_summaries.
    """

    community_id: str
    """Content-addressed over sorted member concept ids — see
    ragcore.concepts.communities.detect_communities. Stable as long as
    community membership doesn't change; changes on the next re-cluster if
    it does, which is what makes summary recomputation resumable/cheap."""

    label: str
    summary: str
    agreements: tuple[str, ...]
    divergences: tuple[str, ...]
    concept_ids: tuple[str, ...]
    source_chunk_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    """Distinct book/document source_ids this community's summary draws on —
    what ClaudeCommunitySummarizer.summarize() sampled across, per the
    round-robin-by-source rule that stops one long book from dominating."""

    model: str
    prompt_version: str
    created_at: str
    """ISO-8601 UTC timestamp string, for display/debugging only — never
    used to invalidate a summary (the community_id itself does that)."""

    vector: tuple[float, ...] | None = None
    """The summary text's own embedding, computed once when the summary is
    built (see build_community_summaries) and persisted alongside it. Lets
    ragcore.retrieval.concept.ConceptRetriever score a summary against a
    query on the same scale as every other retrieval tier, and gives the
    synthetic Chunk built from this summary a real vector for MMR's
    doc-doc similarity — never invented at retrieval time."""

    chunk_fingerprint: str | None = None
    """sha256 of every chunk_id currently reachable from this community's
    concepts, at the time this summary was last computed — distinct from
    `source_chunk_ids` (only the chunks the LLM actually cited, a narrower,
    sampling-dependent set). `community_id` alone only detects a change in
    *which concepts* belong to the community; a community whose membership
    is unchanged but whose underlying chunk text changed (e.g. a source was
    re-ingested) would otherwise keep a stale summary forever. See
    ragcore.concepts.communities.build_community_summaries."""
