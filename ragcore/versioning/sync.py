"""plan_sync / sync_corpus — incremental ingestion: detect new, changed, or
removed documents against an IngestionManifest, and update only the
affected vector store entries, lexical index, entity/concept graph nodes
and edges, hierarchy nodes, and contextual-embedding blurbs.

**A latent characteristic of `Corpus.ingest()` this module exists to work
around, not one it introduces**: `Document.source_id` is a content hash
(sha256 of file bytes). `Corpus.ingest(replace=True)` deletes
`store.delete(source_id=doc.source_id)` using the *newly computed* hash —
so re-ingesting a file whose content changed computes a brand-new
source_id, the delete call removes nothing (nothing exists under that hash
yet), and the file's *old* chunks are silently orphaned in the vector store
forever. This has always been true of a bare `corpus.ingest()` call on a
changed file; it's simply the first time anything in this codebase needed
to track "what did this file's identity used to be," which requires
external state `Corpus` doesn't keep. `sync_corpus` is where this gets
handled correctly, by tracking path -> source_id history in
`IngestionManifest`. A bare `corpus.ingest()` call outside `sync_corpus`
still has this characteristic — see CLAUDE.md.

Order of operations in `sync_corpus`, and why it matters:
1. Snapshot everything (if enabled) — *before* any mutation.
2. Resolve each changed/removed path's *old* source_id from the manifest.
3. Prune every removed/changed source's chunk associations from every
   provided store, *before* any rebuild — a rebuild-then-prune window would
   let corpus-global passes (detect_communities) cluster over a stale union
   of two generations of the same source.
4. Ingest new/changed paths (fresh source_ids), update the manifest.
5. Re-run every LLM-backed pass *scoped to only the new/changed source_ids*
   — build_graph/build_concepts/build_contextual_embeddings/build_hierarchy
   all already support (or, for the two added this session, were built to
   support) a `source_ids` filter, so no new capability was needed here.
6. Re-run the corpus-global concept-community passes unscoped — both are
   already checkpointed and no-op when nothing relevant changed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from ragcore.versioning.manifest import IngestionManifest, hash_file
from ragcore.versioning.types import SyncPlan, SyncReport

if TYPE_CHECKING:
    from ragcore.corpus import Corpus


def plan_sync(manifest: IngestionManifest, paths: Sequence[str | Path]) -> SyncPlan:
    """.added / .changed / .removed / .unchanged, computed by re-hashing
    each given path with the same content-addressing scheme
    ragcore.types.Document.source_id already uses, and comparing against
    the manifest's last-known hash for that path."""
    resolved = {str(Path(p).resolve()): Path(p) for p in paths}
    known = manifest.current_state()

    added: list[str] = []
    changed: list[str] = []
    unchanged: list[str] = []
    for path_str in resolved:
        content_hash = hash_file(path_str)
        known_state = known.get(path_str)
        if known_state is None:
            added.append(path_str)
        elif known_state.content_hash != content_hash:
            changed.append(path_str)
        else:
            unchanged.append(path_str)

    removed = [path_str for path_str in known if path_str not in resolved]

    return SyncPlan(
        added=tuple(added), changed=tuple(changed), removed=tuple(removed), unchanged=tuple(unchanged)
    )


def sync_corpus(
    corpus: Corpus,
    manifest: IngestionManifest,
    paths: Sequence[str | Path],
    *,
    lexical: Any = None,
    graph_store: Any = None,
    graph_extractor: Any = None,
    graph_resolver: Any = None,
    concept_store: Any = None,
    concept_extractor: Any = None,
    concept_resolver: Any = None,
    hierarchy_store: Any = None,
    hierarchy_summarizer: Any = None,
    context_store: Any = None,
    context_writer: Any = None,
    community_summarizer: Any = None,
    snapshot: bool = True,
    snapshot_dir: str | Path | None = None,
) -> SyncReport:
    plan = plan_sync(manifest, paths)
    has_work = bool(plan.added or plan.changed or plan.removed)

    snapshot_tag = None
    if snapshot and has_work:
        from ragcore.versioning.snapshot import create_snapshot

        snapshot_tag = f"sync-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        create_snapshot(
            snapshot_tag,
            manifest,
            corpus.store,
            snapshot_dir=snapshot_dir or Path(".ragcore-snapshots"),
            graph_store=graph_store,
            concept_store=concept_store,
            hierarchy_store=hierarchy_store,
            context_store=context_store,
            lexical=lexical,
        )

    # -- prune removed and (old-identity) changed sources, before any rebuild --
    old_source_ids: list[str] = []
    for path_str in plan.removed:
        old_source_id = manifest.source_id_for_path(path_str)
        if old_source_id is not None:
            old_source_ids.append(old_source_id)
        manifest.mark_removed(path_str)

    for path_str in plan.changed:
        old_source_id = manifest.source_id_for_path(path_str)
        if old_source_id is not None:
            old_source_ids.append(old_source_id)

    for old_source_id in old_source_ids:
        corpus.remove(old_source_id)
        if lexical is not None:
            lexical.delete(source_id=old_source_id)
        if graph_store is not None:
            graph_store.remove_source(old_source_id)
        if concept_store is not None:
            concept_store.remove_source(old_source_id)
        if hierarchy_store is not None:
            hierarchy_store.remove_source(old_source_id)
        if context_store is not None:
            context_store.remove_source(old_source_id)

    # -- ingest new/changed paths under their current (correct) source_id --
    new_or_changed_source_ids: list[str] = []
    chunks_ingested = 0
    errors: list[tuple[str, str]] = []

    for path_str in (*plan.added, *plan.changed):
        try:
            report = corpus.ingest(path_str)
        except Exception as e:  # noqa: BLE001 - collected, not raised, matching Corpus.ingest_many's own convention
            errors.append((path_str, str(e)))
            continue

        source_id = report.source_ids[0]
        new_or_changed_source_ids.append(source_id)
        chunks_ingested += report.chunks_ingested

        if lexical is not None:
            lexical.index(corpus.store.get_by_source(source_id))

        manifest.record_ingest(
            path=path_str, source_id=source_id,
            content_hash=hash_file(path_str), chunk_count=report.chunks_ingested,
        )

    # -- LLM-backed passes, scoped to only the new/changed sources ---------
    if new_or_changed_source_ids:
        if graph_store is not None and graph_extractor is not None:
            from ragcore.graph.pipeline import build_graph

            build_graph(
                corpus, graph_store, graph_extractor, graph_resolver,
                source_ids=new_or_changed_source_ids,
            )

        if concept_store is not None and concept_extractor is not None:
            from ragcore.concepts.pipeline import build_concepts

            build_concepts(
                corpus, concept_store, concept_extractor, concept_resolver,
                source_ids=new_or_changed_source_ids,
            )

        if context_store is not None and context_writer is not None:
            from ragcore.contextual.pipeline import build_contextual_embeddings

            build_contextual_embeddings(
                corpus, context_store, context_writer, source_ids=new_or_changed_source_ids
            )

        if hierarchy_store is not None and hierarchy_summarizer is not None:
            from ragcore.hierarchy.pipeline import build_hierarchy

            build_hierarchy(
                corpus, hierarchy_store, hierarchy_summarizer, source_ids=new_or_changed_source_ids
            )

    # -- corpus-global re-derivation: self-checkpointing, no-ops if unchanged --
    if concept_store is not None and community_summarizer is not None:
        from ragcore.concepts.communities import build_community_summaries, detect_communities

        detect_communities(concept_store)
        build_community_summaries(corpus, concept_store, community_summarizer)

    return SyncReport(
        added=plan.added,
        changed=plan.changed,
        removed=plan.removed,
        unchanged=plan.unchanged,
        chunks_ingested=chunks_ingested,
        errors=tuple(errors),
        snapshot_tag=snapshot_tag,
    )
