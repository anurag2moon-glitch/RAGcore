"""ClaudeEntityExtractor's SYSTEM_PROMPT names the book-tuned entity-type
taxonomy (docs/ARCHITECTURE.md Phase 4 gap list item 16) and explicitly
excludes themes/claims (handled by ragcore.concepts/ragcore.claims
instead, not duplicated here). A regression guard on prompt content, not
a behavioral test — the model's actual choices aren't something a unit
test can verify.
"""

from __future__ import annotations

from ragcore.graph.claude_extractor import SYSTEM_PROMPT


def test_prompt_names_the_book_tuned_entity_types() -> None:
    for entity_type in ["character", "person", "place", "work", "organization", "event"]:
        assert f'"{entity_type}"' in SYSTEM_PROMPT


def test_prompt_explicitly_excludes_themes_and_abstract_propositions() -> None:
    assert "abstract themes" in SYSTEM_PROMPT or "theme" in SYSTEM_PROMPT.lower()
    assert "a separate process handles those" in SYSTEM_PROMPT
