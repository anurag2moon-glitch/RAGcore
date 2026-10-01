"""Community detection and summarization over the concept co-occurrence
graph.

detect_communities() clusters ragcore.concepts.types.ConceptEdge rows with
Louvain by default, or Leiden (`algorithm="leiden"`, docs/ARCHITECTURE.md
Phase 4 gap list item 19) — both via `networkx.algorithms.community` (the
`concepts` extra, lazily imported like every other optional backend in
this codebase; `leiden_communities` needs a recent-enough networkx —
confirmed present in 3.7, the version this pass was developed against).
Louvain stays the default: it's the one already proven in this codebase's
own tests, and Leiden's real advantage over it — guaranteeing every
resulting community is well-connected, which matters most for large,
multi-level hierarchical clustering — isn't yet exercised by anything in
this codebase (`detect_communities()` produces one flat level of
communities either way). `algorithm="leiden"` is available now for a
corpus where that guarantee is worth it; switch the default later if a
real corpus shows Louvain producing disconnected or low-quality
communities Leiden doesn't.
build_community_summaries() then synthesizes, via an LLM, what each
resulting cluster of related concepts collectively says — precomputed and
stored so ragcore.retrieval.concept.ConceptRetriever can surface a summary
directly at query time rather than recomputing it live.

Both are checkpointed independently, at different granularities: community
detection is corpus-global (one clustering run either matches the concept
graph's current state or it doesn't — see the module-level fingerprint
below), while each community's summary is checkpointed individually
(ConceptStore.community_summary), since a single new chapter can shift a few
communities without touching most of them, and re-summarizing every
community on every build_community_summaries() call would be wasteful.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, Sequence

from pydantic import BaseModel

from ragcore.concepts.base import CommunitySummarizer, ConceptStore
from ragcore.concepts.types import CommunitySummary, Concept
from ragcore.errors import GenerationError
from ragcore.graph.claude_extractor import _format_chunks

if TYPE_CHECKING:
    from ragcore.corpus import Corpus
    from ragcore.types import Chunk

try:
    import networkx
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _IMPORT_ERROR = e
    networkx = None
else:
    _IMPORT_ERROR = None

try:
    import anthropic
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    _ANTHROPIC_IMPORT_ERROR = e
    anthropic = None
else:
    _ANTHROPIC_IMPORT_ERROR = None

LOUVAIN_VERSION = "networkx-louvain-v1"
LEIDEN_VERSION = "networkx-leiden-v1"
PROMPT_VERSION = "community-summary-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- community detection -------------------------------------------------


@dataclass(frozen=True)
class CommunityReport:
    ran: bool
    """False when the concept graph's fingerprint matched the last run's —
    detection was skipped entirely, existing community assignments untouched."""

    community_count: int
    concept_count: int


def _run_fingerprint(
    concepts: Sequence[Concept],
    edges: Sequence,
    resolution: float,
    seed: int,
    algorithm_version: str,
) -> str:
    concept_ids = sorted(c.concept_id for c in concepts)
    edge_triples = sorted(f"{e.source_concept_id}|{e.target_concept_id}|{e.weight}" for e in edges)
    payload = "\n".join([
        algorithm_version, f"resolution={resolution}", f"seed={seed}",
        *concept_ids, *edge_triples,
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _community_id(member_concept_ids: Sequence[str]) -> str:
    digest = hashlib.sha256(",".join(sorted(member_concept_ids)).encode("utf-8")).hexdigest()
    return f"community:{digest[:16]}"


def detect_communities(
    store: ConceptStore,
    *,
    algorithm: Literal["louvain", "leiden"] = "louvain",
    resolution: float = 1.0,
    seed: int = 0,
    force: bool = False,
) -> CommunityReport:
    """Clusters the concept graph (Louvain by default, or Leiden —
    `algorithm="leiden"`, see module docstring) and persists concept_id ->
    community_id assignments. A no-op (ran=False) when the concept graph is
    identical to the last successful run under the same algorithm —
    checked via a fingerprint over the algorithm, every concept id, and
    every (source, target, weight) edge triple, so adding/resolving a
    concept, or switching algorithms, both invalidate it and the next call
    reclusters."""
    if networkx is None:
        raise GenerationError(
            "community detection requires the 'concepts' extra: install ragcore[concepts]"
        ) from _IMPORT_ERROR
    if algorithm == "leiden" and not hasattr(networkx.algorithms.community, "leiden_communities"):
        raise GenerationError(
            "algorithm='leiden' requires a networkx version with "
            "networkx.algorithms.community.leiden_communities (present in 3.7+); "
            "upgrade the 'concepts' extra's networkx or use algorithm='louvain'"
        )

    algorithm_version = LEIDEN_VERSION if algorithm == "leiden" else LOUVAIN_VERSION
    concepts = store.all_concepts()
    edges = store.all_edges()
    fingerprint = _run_fingerprint(concepts, edges, resolution, seed, algorithm_version)

    if not force and store.community_run_fingerprint() == fingerprint:
        return CommunityReport(ran=False, community_count=len(store.all_community_ids()), concept_count=len(concepts))

    graph = networkx.Graph()
    graph.add_nodes_from(c.concept_id for c in concepts)
    for e in edges:
        graph.add_edge(e.source_concept_id, e.target_concept_id, weight=e.weight)

    if graph.number_of_nodes() == 0:
        clusters: list[set[str]] = []
    elif algorithm == "leiden":
        clusters = list(
            networkx.algorithms.community.leiden_communities(
                graph, weight="weight", resolution=resolution, seed=seed, metric="modularity"
            )
        )
    else:
        clusters = list(
            networkx.algorithms.community.louvain_communities(
                graph, weight="weight", resolution=resolution, seed=seed
            )
        )

    assignments: dict[str, str] = {}
    for cluster in clusters:
        community_id = _community_id(cluster)
        for concept_id in cluster:
            assignments[concept_id] = community_id

    store.set_communities(run_fingerprint=fingerprint, assignments=assignments)
    # set_communities() replaces assignments but never garbage-collects the
    # previous run's now-vanished communities' summaries — close that leak
    # here, right after a real reclustering (never on the no-op path above,
    # since nothing could have vanished if the graph didn't change).
    store.prune_orphan_summaries()
    return CommunityReport(ran=True, community_count=len(clusters), concept_count=len(concepts))


# -- community summarization ---------------------------------------------


def _sample_chunks_round_robin(corpus: Corpus, chunk_ids: Sequence[str], *, max_chunks: int) -> list[Chunk]:
    """Hydrates chunk_ids and samples up to max_chunks of them, round-robin
    across distinct source_ids — so one long book's chunks (which naturally
    outnumber a short pamphlet's in raw count) don't crowd out every other
    contributing source from the summarization prompt."""
    hydrated = corpus.get(list(dict.fromkeys(chunk_ids)))
    by_source: dict[str, list[Chunk]] = {}
    for c in hydrated:
        by_source.setdefault(c.meta.source_id, []).append(c)
    for group in by_source.values():
        group.sort(key=lambda c: c.meta.chunk_index)

    sampled: list[Chunk] = []
    queues = list(by_source.values())
    i = 0
    while queues and len(sampled) < max_chunks:
        queue = queues[i % len(queues)]
        if queue:
            sampled.append(queue.pop(0))
        queues = [q for q in queues if q]
        i += 1
        if not queues:
            break
    return sampled[:max_chunks]


def _chunk_fingerprint(chunk_ids: Sequence[str]) -> str:
    return hashlib.sha256(",".join(sorted(set(chunk_ids))).encode("utf-8")).hexdigest()


def build_community_summaries(
    corpus: Corpus,
    store: ConceptStore,
    summarizer: CommunitySummarizer,
    *,
    force: bool = False,
    max_chunks_per_community: int = 20,
) -> int:
    """Synthesizes and stores a CommunitySummary for every community lacking
    one, or whose full set of reachable chunk_ids has changed since its
    summary was last computed (or every community, with force=True).
    Returns the number of summaries (re)computed.

    Staleness is checked against `chunk_fingerprint` — a hash of *every*
    chunk_id currently reachable from the community's concepts — not
    `source_chunk_ids` (only the chunks the LLM actually cited, a narrower,
    sampling-dependent set that would almost always differ between runs
    even when nothing real changed). community_id alone only detects a
    change in *which concepts* belong to the community; a community whose
    membership is unchanged but whose underlying chunk text changed (e.g. a
    source was re-ingested) would otherwise keep a stale summary forever —
    see ragcore.concepts.types.CommunitySummary.chunk_fingerprint.
    """
    computed = 0
    for community_id in store.all_community_ids():
        concept_ids = store.concept_ids_for_community(community_id)
        concepts = [c for c in (store.concept(cid) for cid in concept_ids) if c is not None]
        if not concepts:
            continue

        chunk_ids_by_concept = store.chunks_for_concepts(concept_ids)
        all_chunk_ids = [cid for ids in chunk_ids_by_concept.values() for cid in ids]
        fingerprint = _chunk_fingerprint(all_chunk_ids)

        existing = store.community_summary(community_id)
        if not force and existing is not None and existing.chunk_fingerprint == fingerprint:
            continue

        chunks = _sample_chunks_round_robin(corpus, all_chunk_ids, max_chunks=max_chunks_per_community)
        if not chunks:
            continue

        summary = summarizer.summarize(community_id=community_id, concepts=concepts, chunks=chunks)
        summary = dataclasses.replace(summary, chunk_fingerprint=fingerprint)

        # The summary's own embedding is computed here (not by the
        # summarizer) so a CommunitySummarizer implementation never needs an
        # Embedder dependency of its own — it stays scoped to "synthesize
        # text", matching how ClaudeCommunitySummarizer only needs anthropic.
        if summary.vector is None:
            vec = corpus.embedder.embed_groups([[summary.summary]], input_type="document")[0][0]
            summary = dataclasses.replace(summary, vector=tuple(vec))

        store.put_community_summary(summary)
        computed += 1

    return computed


SYSTEM_PROMPT = (
    "You write a synthesized summary of one cluster of related abstract "
    "concepts/themes drawn from one or more books, for a reader who wants "
    "to understand what these sources collectively say about this cluster "
    "of ideas — without reading every book.\n\n"
    "You are given the cluster's concept labels and descriptions, then a "
    "set of source excerpts (each numbered in brackets, like [1]) that "
    "discuss them. Write: a short label for the cluster as a whole; a "
    "summary (a few sentences) of what these sources collectively say about "
    "it; explicit points of agreement across sources (as separate short "
    "bullet strings); explicit points of divergence or disagreement across "
    "sources (also separate short bullet strings — an empty list if the "
    "sources agree throughout, don't invent disagreement that isn't there); "
    "and the bracketed number(s) of every excerpt that actually contributed "
    "to this summary (e.g. 1, 3 — not the text \"[1]\", just the number).\n\n"
    "Ground every sentence in the given excerpts. Do not use outside "
    "knowledge about the books or their themes."
)


class _CommunitySummaryResult(BaseModel):
    label: str
    summary: str
    points_of_agreement: list[str] = []
    points_of_divergence: list[str] = []
    contributing_chunk_indices: list[int]


class ClaudeCommunitySummarizer:
    """CommunitySummarizer implementation using Claude (claude-opus-5 by
    default). Same messages.parse / retry / error-wrapping shape as every
    other Claude-backed component in this codebase."""

    def __init__(
        self,
        *,
        model: str = "claude-opus-5",
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 8192,
        max_retries: int = 5,
    ) -> None:
        if client is None and anthropic is None:
            raise GenerationError(
                "Claude community summarization requires the 'generation' extra: "
                "install ragcore[generation]"
            ) from _ANTHROPIC_IMPORT_ERROR

        self.model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._client = client or (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def summarize(
        self, *, community_id: str, concepts: Sequence[Concept], chunks: Sequence[Chunk]
    ) -> CommunitySummary:
        concept_list = "\n".join(f"- {c.label}: {c.description}" for c in concepts)
        formatted, index_to_chunk_id = _format_chunks(chunks)
        user_content = f"Concepts in this cluster:\n{concept_list}\n\nSource excerpts:\n{formatted}"

        parsed = self._call_with_retry(user_content)

        cited_chunk_ids = tuple(
            index_to_chunk_id[i] for i in parsed.contributing_chunk_indices if i in index_to_chunk_id
        )
        # A response that cited nothing resolvable still needs *some*
        # provenance — fall back to every chunk we actually showed it rather
        # than persisting a summary with no source_chunk_ids at all.
        source_chunk_ids = cited_chunk_ids or tuple(c.chunk_id for c in chunks)
        contributing = [c for c in chunks if c.chunk_id in source_chunk_ids] or list(chunks)
        source_ids = tuple(sorted({c.meta.source_id for c in contributing}))

        return CommunitySummary(
            community_id=community_id,
            label=parsed.label,
            summary=parsed.summary,
            agreements=tuple(parsed.points_of_agreement),
            divergences=tuple(parsed.points_of_divergence),
            concept_ids=tuple(c.concept_id for c in concepts),
            source_chunk_ids=source_chunk_ids,
            source_ids=source_ids,
            model=self.model,
            prompt_version=PROMPT_VERSION,
            created_at=_now(),
        )

    def _call_with_retry(self, user_content: str) -> _CommunitySummaryResult:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.parse(
                    model=self.model,
                    max_tokens=self._max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                    output_format=_CommunitySummaryResult,
                )
                return response.parsed_output
            except anthropic.RateLimitError as e:
                last_err = e
            except anthropic.APIConnectionError as e:
                last_err = e
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    last_err = e
                else:
                    raise GenerationError(f"{self.model} community summarization request rejected: {e}") from e

            if attempt < self._max_retries - 1:
                time.sleep(min(2**attempt, 30))

        raise GenerationError(
            f"{self.model} community summarization failed after {self._max_retries} attempts: {last_err}"
        ) from last_err
