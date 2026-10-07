"""Lal Kitab astrology domain pack for ragcore.rules (Phase 10 §7) —
the first concrete DomainPack, proving the generic rule layer in
ragcore.rules needs no core changes to add a domain. See
docs/adr/0013-rule-extraction-domain-pluggable-architecture.md.
"""

from lal_kitab.ontology import LalKitabOntologyNormalizer
from lal_kitab.pack import LalKitabPack

__all__ = ["LalKitabOntologyNormalizer", "LalKitabPack"]
