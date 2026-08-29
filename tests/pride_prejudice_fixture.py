"""Hand-authored fixture mimicking what per-chapter Claude extraction would
produce for Pride and Prejudice Chapters I-III — grounded in the real,
already-ingested chunks in samples/library.lancedb (no fabricated content;
every entity/relation below is directly textually supported by the chunk(s)
it cites). Exists so the graph construction/resolution/provenance pipeline
can be exercised end to end with zero API calls: no ANTHROPIC_API_KEY (no
real extraction), no VOYAGE_API_KEY (StubEmbedder stands in for the real
embedder that EmbeddingEntityResolver would otherwise use).

Deliberately encodes cross-chapter naming drift, the same way independent
per-chapter LLM extraction calls would produce it:
  - Chapter I candidate "Bingley"  vs  Chapter II/III candidate "Mr. Bingley"
  - Chapter II candidate "Elizabeth"  vs  Chapter III candidate "Elizabeth Bennet"
  - Chapter III candidate "the eldest Miss Bennet" (Darcy/Bingley's ballroom
    exchange never uses "Jane" — that name only appears back in Chapter I)
StubEmbedder is tuned so the first two merge (they're genuinely close by any
text-similarity measure) and the third deliberately does not — a real,
honest limitation of a purely text-embedding-based resolver: it has no way
to know "the eldest Miss Bennet" and "Jane" are the same person without
either wider context or relational (not just textual) reasoning. See
CLAUDE.md and the graph-demo output for the fuller writeup.
"""

from __future__ import annotations

import re
from typing import Sequence

from ragcore.graph.resolver import EmbeddingEntityResolver
from ragcore.graph.types import Entity, Relation
from ragcore.types import Chunk

SRC = "462be7852d84412c6695851395144a97e9762d45bd3c41b9f356dc7ac047b8a9"


def cid(n: int) -> str:
    return f"{SRC}:{n}"


def _entity_id(entity_type: str, name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return f"entity:{entity_type}:{slug}"


def _entity(
    name: str, entity_type: str, chunk_ids: tuple[str, ...], aliases: list[str] | None = None
) -> Entity:
    return Entity(
        entity_id=_entity_id(entity_type, name),
        name=name,
        entity_type=entity_type,
        source_chunk_ids=chunk_ids,
        extra={"aliases": aliases} if aliases else {},
    )


def _relation(rel_type: str, source: Entity, target: Entity, chunk_ids: tuple[str, ...]) -> Relation:
    return Relation(
        relation_id=f"relation:{rel_type}:{source.entity_id}:{target.entity_id}",
        source_entity_id=source.entity_id,
        target_entity_id=target.entity_id,
        relation_type=rel_type,
        source_chunk_ids=chunk_ids,
    )


def chapter_1_candidates() -> tuple[list[Entity], list[Relation]]:
    """Chunks 24 + 26 (Chapter I, chapter_index=2). Grounding:
    chunk 24: "What is his name?" / "Bingley." ; "Netherfield is taken by a
    young man of large fortune... Bingley."
    chunk 26: "...not half so handsome as Jane, nor half so good-humoured
    as Lydia."
    """
    mr_bennet = _entity("Mr. Bennet", "person", (cid(24),), ["her husband"])
    mrs_bennet = _entity("Mrs. Bennet", "person", (cid(24),), ["his lady", "his wife"])
    bingley = _entity("Bingley", "person", (cid(24),))
    jane = _entity("Jane", "person", (cid(26),))
    lydia = _entity("Lydia", "person", (cid(26),))
    netherfield = _entity("Netherfield Park", "place", (cid(24),))

    entities = [mr_bennet, mrs_bennet, bingley, jane, lydia, netherfield]
    relations = [
        _relation("spouse_of", mr_bennet, mrs_bennet, (cid(24),)),
        _relation("resides_at", bingley, netherfield, (cid(24),)),
    ]
    return entities, relations


def chapter_2_candidates() -> tuple[list[Entity], list[Relation]]:
    """Chunk 28 (Chapter II, chapter_index=3). Grounding: "Mr. Bennet was
    among the earliest of those who waited on Mr. Bingley"; "Observing his
    second daughter..." (Elizabeth, addressed as "Lizzy"); "Kitty has no
    discretion in her coughs," said her father; "Mrs. Long has promised to
    introduce him."
    """
    mr_bennet = _entity("Mr. Bennet", "person", (cid(28),), ["her father"])
    mrs_bennet = _entity("Mrs. Bennet", "person", (cid(28),), ["her mother"])
    mr_bingley = _entity("Mr. Bingley", "person", (cid(28),))  # deliberately != ch1's "Bingley"
    elizabeth = _entity("Elizabeth", "person", (cid(28),), ["Lizzy", "his second daughter"])
    kitty = _entity("Kitty", "person", (cid(28),))
    mrs_long = _entity("Mrs. Long", "person", (cid(28),))

    entities = [mr_bennet, mrs_bennet, mr_bingley, elizabeth, kitty, mrs_long]
    relations = [
        _relation("acquainted_with", mr_bennet, mr_bingley, (cid(28),)),
        _relation("parent_of", mr_bennet, elizabeth, (cid(28),)),
        _relation("parent_of", mr_bennet, kitty, (cid(28),)),
        _relation("acquainted_with", mrs_long, mr_bingley, (cid(28),)),
    ]
    return entities, relations


def chapter_3_candidates() -> tuple[list[Entity], list[Relation]]:
    """Chunks 35 + 36 (Chapter III, chapter_index=4). Grounding: "What a
    contrast between him and his friend! Mr. Darcy..."; "Mr. Darcy danced
    only once with Mrs. Hurst and once with Miss Bingley"; "Elizabeth
    Bennet had been obliged..."; "...his having slighted one of her
    daughters" (Mrs. Bennet's daughter); "You are dancing with the only
    handsome girl in the room," said Mr. Darcy, looking at the eldest Miss
    Bennet — chunk 36 never uses the name "Jane".
    """
    mr_bingley = _entity("Mr. Bingley", "person", (cid(35), cid(36)))  # matches ch2's string exactly
    mr_darcy = _entity("Mr. Darcy", "person", (cid(35), cid(36)), ["his friend"])
    eldest_miss_bennet = _entity("the eldest Miss Bennet", "person", (cid(36),))
    elizabeth_bennet = _entity("Elizabeth Bennet", "person", (cid(35),))  # deliberately != ch2's "Elizabeth"
    mrs_hurst = _entity("Mrs. Hurst", "person", (cid(35),))
    miss_bingley = _entity("Miss Bingley", "person", (cid(35),))
    mrs_bennet = _entity("Mrs. Bennet", "person", (cid(35),))  # exact match to existing — no resolver needed

    entities = [
        mr_bingley, mr_darcy, eldest_miss_bennet, elizabeth_bennet, mrs_hurst, miss_bingley, mrs_bennet,
    ]
    relations = [
        _relation("friend_of", mr_darcy, mr_bingley, (cid(35),)),
        _relation("danced_with", mr_darcy, mrs_hurst, (cid(35),)),
        _relation("danced_with", mr_darcy, miss_bingley, (cid(35),)),
        _relation("slighted", mr_darcy, elizabeth_bennet, (cid(35), cid(36))),
        _relation("parent_of", mrs_bennet, elizabeth_bennet, (cid(35),)),
        _relation("danced_with", mr_bingley, eldest_miss_bennet, (cid(36),)),
    ]
    return entities, relations


CANDIDATES_BY_CHAPTER_INDEX = {
    2: chapter_1_candidates,
    3: chapter_2_candidates,
    4: chapter_3_candidates,
}


def extractor_fn(chunks: Sequence[Chunk]) -> tuple[list[Entity], list[Relation]]:
    """The callable FakeEntityExtractor is built around: dispatches on the
    chapter_index of the chunks it's given (build_graph groups by chapter
    before calling extract()), returns ([], []) for any chapter this
    fixture doesn't cover — the real corpus has 7 ingested chapters, this
    fixture only hand-authors the first 3.
    """
    chapter_index = chunks[0].meta.chapter_index
    fn = CANDIDATES_BY_CHAPTER_INDEX.get(chapter_index)
    return fn() if fn else ([], [])


class StubEmbedder:
    """Controllable stand-in for the real Embedder: assigns each distinct
    real-world identity its own one-hot axis (via substring match on known
    name markers, checked longest/most-specific first — e.g. "miss bingley"
    before "bingley", so Mr. Bingley's *sister* never gets confused with Mr.
    Bingley himself), so EmbeddingEntityResolver's cosine similarity is
    exactly what this fixture sets it to be:

      - "Bingley" and "Mr. Bingley" map to the *same* axis (identical
        vectors, cosine=1.0) -> should merge.
      - "Elizabeth" and "Elizabeth Bennet" likewise share an axis -> merge.
      - "the eldest Miss Bennet" gets its own axis, orthogonal to "Jane"'s
        (cosine=0) -> should NOT merge — seeded a bug earlier this way: a
        single shared *fallback* vector for every unmapped identity made
        every stranger look identical to every other stranger and merged
        nearly the whole cast into one entity. Every distinct person here
        now gets an explicit, mutually orthogonal axis instead.
    """

    name = "fixture-stub"
    dim = 13

    _AXES = [
        "eldest_miss_bennet", "elizabeth", "mrs_hurst", "miss_bingley", "bingley",
        "mr_bennet", "mrs_bennet", "darcy", "jane", "lydia", "kitty", "mrs_long",
        "netherfield_park",
    ]
    # Longest/most-specific phrase first: substring matching must not let a
    # shorter marker ("bingley") shadow a longer one ("miss bingley").
    _RULES = [
        ("eldest miss bennet", "eldest_miss_bennet"),
        ("elizabeth", "elizabeth"),
        ("mrs. hurst", "mrs_hurst"),
        ("miss bingley", "miss_bingley"),
        ("bingley", "bingley"),
        ("mr. bennet", "mr_bennet"),
        ("mrs. bennet", "mrs_bennet"),
        ("darcy", "darcy"),
        ("jane", "jane"),
        ("lydia", "lydia"),
        ("kitty", "kitty"),
        ("mrs. long", "mrs_long"),
        ("netherfield", "netherfield_park"),
    ]

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        for phrase, axis in self._RULES:
            if phrase in lowered:
                v = [0.0] * len(self._AXES)
                v[self._AXES.index(axis)] = 1.0
                return v
        raise KeyError(f"StubEmbedder: no identity rule matches {text!r} — fixture vocabulary is fixed")

    def embed_groups(self, groups, *, input_type="document"):
        return [[self._vector(text) for text in group] for group in groups]

    def embed_query(self, text):
        return self._vector(text)


def build_resolver() -> EmbeddingEntityResolver:
    return EmbeddingEntityResolver(StubEmbedder(), threshold=0.9)
