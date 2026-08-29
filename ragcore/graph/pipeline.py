"""build_graph — the construction pass that turns already-ingested chunks
into a populated GraphStore.

A separate pass, not part of Corpus.ingest(): it reads chunks back out of
the vector store (VectorStore.get_by_source), groups them by chapter
(context_group_id) so EntityExtractor gets chapter-level context, resolves
each candidate entity against what's already in the GraphStore, and writes
the result. Corpus.ingest() is never touched by this — see the design
doc's graph-attachment principle.

Content-addressed entity/relation ids (ragcore.graph.claude_extractor)
make this idempotent: re-running it over the same source just re-merges
into the same rows (SqliteGraphStore.add_entities/add_relations merge
rather than replace).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from ragcore.graph.base import EntityExtractor, GraphStore
from ragcore.graph.resolver import EmbeddingEntityResolver, EntityResolver
from ragcore.graph.types import Entity, Relation

if TYPE_CHECKING:
    # Deferred to avoid a cycle: ragcore.graph.__init__ imports this module,
    # and ragcore.types imports ragcore.graph.types (for GraphContext) —
    # importing ragcore.corpus or Chunk (both from/via ragcore.types) at
    # module level here would reach back into a not-yet-finished
    # ragcore.types import. build_graph() only ever duck-types `corpus` and
    # the chunks it's given, so neither real class is needed at runtime.
    from ragcore.corpus import Corpus
    from ragcore.types import Chunk


@dataclass(frozen=True)
class GraphReport:
    chunks_processed: int
    entities_extracted: int
    relations_extracted: int
    entities_merged: int


def _group_by_context_group(chunks: Sequence[Chunk]) -> list[list[Chunk]]:
    """Group chunks (already ordered by chunk_index) into their chapters.
    Each ChunkGroup's chunks are contiguous in chunk_index order by
    construction (see StructuralChunker), so a single linear pass suffices.
    """
    groups: dict[str, list[Chunk]] = {}
    order: list[str] = []
    for c in chunks:
        gid = c.meta.context_group_id
        if gid not in groups:
            groups[gid] = []
            order.append(gid)
        groups[gid].append(c)
    return [groups[gid] for gid in order]


def build_graph(
    corpus: Corpus,
    store: GraphStore,
    extractor: EntityExtractor,
    resolver: EntityResolver | None = None,
    *,
    source_ids: Sequence[str] | None = None,
) -> GraphReport:
    resolver = resolver or EmbeddingEntityResolver(corpus.embedder)

    sources = corpus.sources()
    if source_ids is not None:
        wanted = set(source_ids)
        sources = [s for s in sources if s.source_id in wanted]

    chunks_processed = 0
    entities_extracted = 0
    relations_extracted = 0
    entities_merged = 0

    for source in sources:
        chunks = corpus.store.get_by_source(source.source_id)
        for chapter_chunks in _group_by_context_group(chunks):
            candidate_entities, candidate_relations = extractor.extract(chapter_chunks)
            chunks_processed += len(chapter_chunks)

            # Cheap resolution aid: give each candidate a short summary of
            # its same-batch relations (e.g. "danced_with Mr. Bingley"), so
            # EmbeddingEntityResolver has shared relational vocabulary to
            # compare against, not just name/aliases — helps exactly (and
            # only) entities introduced by relational description rather
            # than name ("the eldest Miss Bennet"). Persisted into
            # extra["relations"] on storage, and GraphStore implementations
            # accumulate it across merges the same way they do aliases. See
            # CLAUDE.md's "Known limitation: relationally-described
            # entities" for what this does and doesn't solve.
            id_to_name = {e.entity_id: e.name for e in candidate_entities}
            relation_hints: dict[str, list[str]] = {}
            for rel in candidate_relations:
                src_name = id_to_name.get(rel.source_entity_id, rel.source_entity_id)
                tgt_name = id_to_name.get(rel.target_entity_id, rel.target_entity_id)
                relation_hints.setdefault(rel.source_entity_id, []).append(f"{rel.relation_type} {tgt_name}")
                relation_hints.setdefault(rel.target_entity_id, []).append(f"{rel.relation_type} {src_name}")

            id_remap: dict[str, str] = {}
            resolved_entities: list[Entity] = []
            for candidate in candidate_entities:
                hints = relation_hints.get(candidate.entity_id)
                if hints:
                    candidate = dataclasses.replace(
                        candidate, extra={**candidate.extra, "relations": hints}
                    )

                existing_same_type = store.entities_by_type(candidate.entity_type)
                existing_ids = {e.entity_id for e in existing_same_type}

                if candidate.entity_id in existing_ids:
                    # Exact content-addressed match — add_entities() merges
                    # provenance naturally; no need to consult the resolver
                    # (and no embedding call spent on an unambiguous case).
                    final_id = candidate.entity_id
                else:
                    merge_into = resolver.resolve(candidate, existing_same_type)
                    final_id = merge_into or candidate.entity_id
                    if merge_into is not None:
                        entities_merged += 1

                id_remap[candidate.entity_id] = final_id
                resolved_entities.append(dataclasses.replace(candidate, entity_id=final_id))
                entities_extracted += 1

            # Relations reference the *pre-resolution* candidate entity ids
            # (assigned by the extractor within this same call) — remap them
            # to whatever those candidates actually resolved to, and
            # regenerate relation_id to match (it's derived from endpoint ids).
            remapped_relations: list[Relation] = []
            for rel in candidate_relations:
                source_id = id_remap.get(rel.source_entity_id, rel.source_entity_id)
                target_id = id_remap.get(rel.target_entity_id, rel.target_entity_id)
                remapped_relations.append(
                    dataclasses.replace(
                        rel,
                        source_entity_id=source_id,
                        target_entity_id=target_id,
                        relation_id=f"relation:{rel.relation_type}:{source_id}:{target_id}",
                    )
                )

            store.add_entities(resolved_entities)
            store.add_relations(remapped_relations)
            relations_extracted += len(remapped_relations)

    return GraphReport(
        chunks_processed=chunks_processed,
        entities_extracted=entities_extracted,
        relations_extracted=relations_extracted,
        entities_merged=entities_merged,
    )
