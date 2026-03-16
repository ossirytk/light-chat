"""Shared types and constants for RAG text analysis and enrichment."""

from dataclasses import dataclass
from typing import Any


@dataclass
class TextAnalysisResult:
    """Results from text analysis."""

    total_chars: int
    total_lines: int
    total_words: int
    unique_words: int
    named_entities: list[str]
    key_phrases: list[str]
    potential_metadata: list[dict[str, Any]]
    enrichment_review: list[dict[str, Any]]
    statistics: dict[str, Any]


@dataclass(frozen=True)
class EnrichmentOptions:
    """Options controlling metadata enrichment behavior."""

    auto_categories: bool = True
    auto_aliases: bool = True
    max_aliases: int = 5
    strict: bool = False
    category_confidence_threshold: float = 0.75
    allow_unassigned_categories: bool = False


STOPWORD_ALIASES = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
}

LOCATION_HINTS = {
    "station",
    "deck",
    "bridge",
    "level",
    "bay",
    "grove",
    "operations",
    "engineering",
    "reactor",
    "core",
    "planet",
    "earth",
}

FACTION_HINTS = {
    "corporation",
    "corp",
    "optimum",
    "unn",
    "security",
    "resistance",
    "executives",
}

TECH_HINTS = {
    "protocol",
    "interface",
    "laser",
    "cpu",
    "network",
    "system",
    "cyberspace",
    "mutagen",
    "virus",
    "ai",
}

EVENT_HINTS = {
    "incident",
    "takeover",
    "attack",
    "defeat",
    "escape",
    "operation",
    "conference",
}

CATEGORY_STRICT_THRESHOLD = 0.75
ALIAS_STRICT_THRESHOLD = 0.8
MIN_MULTIWORD_TOKENS = 2
MIN_ACRONYM_LENGTH = 2
MIN_ALIAS_LENGTH = 3


__all__ = [
    "ALIAS_STRICT_THRESHOLD",
    "CATEGORY_STRICT_THRESHOLD",
    "EVENT_HINTS",
    "FACTION_HINTS",
    "LOCATION_HINTS",
    "MIN_ACRONYM_LENGTH",
    "MIN_ALIAS_LENGTH",
    "MIN_MULTIWORD_TOKENS",
    "STOPWORD_ALIASES",
    "TECH_HINTS",
    "EnrichmentOptions",
    "TextAnalysisResult",
]
