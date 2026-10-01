"""merge_extra — shared entity/relation `extra` dict merge for every
GraphStore implementation that merges-not-replaces on write (see
SqliteGraphStore's module docstring for why: the same fact is often
restated across chapters, and every restatement's chunk should accumulate
as provenance rather than the latest write clobbering earlier aliases/
relation hints).

Extracted from SqliteGraphStore once KuzuGraphStore needed the identical
merge logic — both stores read the existing row before writing and merge
in Python, since a JSON-string column's semantic dict merge (as opposed
to a plain list union, which both backends' query languages can do
server-side) has no server-side equivalent in either SQLite or Kuzu.
"""

from __future__ import annotations

LIST_MERGE_KEYS = ("aliases", "relations")
"""extra fields that accumulate (set-union) across merges rather than
being overwritten by the latest write — aliases and relation hints both
grow monotonically as more chapters mention the same entity. See
ragcore.graph.resolver._entity_text for why "relations" exists at all."""


def merge_extra(old: dict, new: dict) -> dict:
    merged = dict(old)
    merged.update(new)
    for key in LIST_MERGE_KEYS:
        combined = set(old.get(key) or []) | set(new.get(key) or [])
        if combined:
            merged[key] = sorted(combined)
    return merged
