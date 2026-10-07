# ragcore-lal-kitab

The Lal Kitab astrology domain pack for `ragcore.rules` (Phase 10 §7)
— the first concrete `DomainPack`, proving the generic rule layer in
`ragcore.rules` needs no core changes to add a domain. See
[docs/adr/0013-rule-extraction-domain-pluggable-architecture.md](../../docs/adr/0013-rule-extraction-domain-pluggable-architecture.md).

A sibling package, not part of `ragcore` itself — see
[docs/adr/0008-api-server-as-sibling-package.md](../../docs/adr/0008-api-server-as-sibling-package.md)
for why concrete, deployable/domain-specific artifacts live outside
`ragcore`'s own dependency tree.

## Contents

- `lal_kitab.vocabulary` — the Navagraha (nine planets), twelve houses
  (by Hindi/Sanskrit ordinal name, e.g. "ashtam bhava" for the 8th),
  twelve rashis (signs), and dignity terms, with every alias/
  transliteration this pack recognizes.
- `lal_kitab.ontology.LalKitabOntologyNormalizer` — maps a raw
  source-text mention to the canonical fact key/value
  `ragcore.rules.evaluator` matches against.
- `lal_kitab.pack.LalKitabPack` — the `DomainPack` implementation:
  `fact_types()` (generated from the vocabulary, not hand-enumerated),
  `extraction_system_prompt()`, `verification_system_prompt()`.

## Local dev

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e "../..[generation]" -e ".[dev]"
pytest -q
```
