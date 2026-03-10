"""Entity extraction and metadata enrichment helpers for RAG text analysis."""

import re
import uuid
from collections import Counter
from typing import Any

from scripts.rag.analyze_rag_text_types import (
    ALIAS_STRICT_THRESHOLD,
    CATEGORY_STRICT_THRESHOLD,
    EVENT_HINTS,
    FACTION_HINTS,
    LOCATION_HINTS,
    MIN_ACRONYM_LENGTH,
    MIN_ALIAS_LENGTH,
    MIN_MULTIWORD_TOKENS,
    STOPWORD_ALIASES,
    TECH_HINTS,
    EnrichmentOptions,
)


def extract_named_entities(text: str) -> list[str]:
    """Extract potential named entities using heuristics."""
    entities = []

    capitalized_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
    capitalized_matches = re.findall(capitalized_pattern, text)
    entities.extend(capitalized_matches)

    quoted_pattern = r'"([^"]+)"'
    quoted_matches = re.findall(quoted_pattern, text)
    entities.extend(quoted_matches)

    date_pattern = r"\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    date_matches = re.findall(date_pattern, text)
    entities.extend(date_matches)

    return list(set(entities))


def extract_key_phrases(text: str, min_freq: int = 3) -> list[str]:
    """Extract key phrases that appear frequently in text."""
    words = re.findall(r"\b\w+\b", text.lower())

    bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
    trigrams = [f"{words[i]} {words[i + 1]} {words[i + 2]}" for i in range(len(words) - 2)]

    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)

    key_bigrams = [phrase for phrase, count in bigram_counts.items() if count >= min_freq]
    key_trigrams = [phrase for phrase, count in trigram_counts.items() if count >= min_freq]

    return key_trigrams[:20] + key_bigrams[:30]


def _extract_context_windows(text: str, entity: str, radius: int = 80) -> list[str]:
    """Extract short lowercase context windows around entity mentions."""
    windows: list[str] = []
    escaped_entity = re.escape(entity)
    for match in re.finditer(escaped_entity, text, flags=re.IGNORECASE):
        start = max(0, match.start() - radius)
        end = min(len(text), match.end() + radius)
        windows.append(text[start:end].lower())
    return windows


def infer_category_with_confidence(entity: str, text: str) -> tuple[str, float]:
    """Infer metadata category with a heuristic confidence score."""
    entity_stripped = entity.strip()
    entity_lower = entity_stripped.lower()
    inferred_category = "concept"
    confidence = 0.5

    if re.fullmatch(r"\d{4}", entity_stripped) or re.search(r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b", entity_stripped):
        inferred_category = "date"
        confidence = 0.98
    else:
        contexts = _extract_context_windows(text, entity_stripped)
        context_text = " ".join(contexts)

        has_location_hint_in_entity = any(hint in entity_lower for hint in LOCATION_HINTS)
        has_tech_hint_in_entity = any(hint in entity_lower for hint in TECH_HINTS)
        has_faction_hint_in_entity = any(hint in entity_lower for hint in FACTION_HINTS)
        has_faction_hint_in_context = any(hint in context_text for hint in FACTION_HINTS)

        if has_tech_hint_in_entity:
            inferred_category = "technology"
            confidence = 0.9
        elif has_faction_hint_in_entity or (has_faction_hint_in_context and not has_location_hint_in_entity):
            inferred_category = "faction"
            confidence = 0.9 if has_faction_hint_in_entity else 0.78
        elif has_location_hint_in_entity or any(hint in context_text for hint in LOCATION_HINTS):
            inferred_category = "location"
            confidence = 0.9 if has_location_hint_in_entity else 0.78
        elif any(hint in entity_lower for hint in EVENT_HINTS) or any(hint in context_text for hint in EVENT_HINTS):
            inferred_category = "event"
            confidence = 0.86 if any(hint in entity_lower for hint in EVENT_HINTS) else 0.74
        elif any(hint in entity_lower for hint in TECH_HINTS) or any(hint in context_text for hint in TECH_HINTS):
            inferred_category = "technology"
            confidence = 0.85 if any(hint in entity_lower for hint in TECH_HINTS) else 0.72
        elif re.fullmatch(r"[A-Z]{3,}", entity_stripped):
            inferred_category = "technology"
            confidence = 0.83
        elif len(entity_stripped.split()) >= MIN_MULTIWORD_TOKENS and re.search(
            r"\b(dr\.?|vice president|researcher|captain)\b",
            context_text,
        ):
            inferred_category = "character"
            confidence = 0.84
        elif len(entity_stripped.split()) >= MIN_MULTIWORD_TOKENS and re.match(
            r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$",
            entity_stripped,
        ):
            inferred_category = "character"
            confidence = 0.73

    return inferred_category, confidence


def infer_category_for_entity(entity: str, text: str) -> str:
    """Infer metadata category for an entity using lightweight heuristics."""
    category, _confidence = infer_category_with_confidence(entity, text)
    return category


def _split_camel_words(text: str) -> str:
    """Convert CamelCase to spaced words."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", text)


def _extract_parenthetical_aliases(entity: str, text: str) -> list[str]:
    """Extract aliases from parenthetical forms around canonical entity mentions."""
    aliases: list[str] = []
    escaped_entity = re.escape(entity)
    patterns = [
        rf"{escaped_entity}\s*\(([^)]+)\)",
        rf"([A-Za-z0-9 .\-]{{2,40}})\s*\({escaped_entity}\)",
        rf"{escaped_entity}\s*,\s*also known as\s+([A-Za-z0-9 .\-]{{2,40}})",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            if isinstance(match, tuple):
                aliases.extend([part for part in match if part])
            else:
                aliases.append(match)
    return [alias.strip() for alias in aliases if alias.strip()]


def _generate_alias_candidates_with_confidence(entity: str, text: str) -> list[tuple[str, float]]:
    """Generate alias candidates with confidence scores."""
    candidates: list[tuple[str, float]] = []
    canonical = entity.strip()
    parenthetical_aliases = _extract_parenthetical_aliases(canonical, text)
    candidates.extend((alias, 0.95) for alias in parenthetical_aliases)

    if len(canonical.split()) >= MIN_MULTIWORD_TOKENS:
        acronym = "".join(word[0].upper() for word in canonical.split() if word and word[0].isalpha())
        if len(acronym) >= MIN_ACRONYM_LENGTH and re.search(rf"\b{re.escape(acronym)}\b", text):
            candidates.append((acronym, 0.92))

    if "-" in canonical:
        candidates.append((canonical.replace("-", ""), 0.65))
        candidates.append((canonical.replace("-", " "), 0.68))
    if " " in canonical:
        candidates.append((canonical.replace(" ", ""), 0.62))

    camel_split = _split_camel_words(canonical)
    if " " not in canonical and "-" not in canonical and camel_split != canonical:
        candidates.append((camel_split, 0.66))

    return candidates


def _select_aliases_with_review(
    candidates: list[tuple[str, float]],
    canonical: str,
    *,
    strict: bool,
    max_aliases: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Select aliases from candidates and provide per-candidate review details."""
    canonical_lower = canonical.lower()
    selected_aliases: list[str] = []
    selected_aliases_lower: set[str] = set()
    review: list[dict[str, Any]] = []

    for candidate, confidence in candidates:
        normalized = re.sub(r"\s+", " ", candidate).strip(" .,-")
        kept = True
        reason = "kept"

        if strict and confidence < ALIAS_STRICT_THRESHOLD:
            kept = False
            reason = "strict_low_confidence"
        elif not normalized:
            kept = False
            reason = "empty_after_normalization"
        elif normalized.lower() == canonical_lower:
            kept = False
            reason = "same_as_canonical"
        elif normalized.lower() in STOPWORD_ALIASES:
            kept = False
            reason = "stopword"
        elif len(normalized) < MIN_ALIAS_LENGTH and not normalized.isupper():
            kept = False
            reason = "too_short"
        elif normalized.lower() in selected_aliases_lower:
            kept = False
            reason = "duplicate"
        elif len(selected_aliases) >= max_aliases:
            kept = False
            reason = "max_aliases_limit"

        if kept:
            selected_aliases.append(normalized)
            selected_aliases_lower.add(normalized.lower())

        review.append(
            {
                "candidate": candidate,
                "normalized": normalized,
                "confidence": round(confidence, 3),
                "kept": kept,
                "reason": reason,
            }
        )

    return selected_aliases, review


def generate_aliases_for_entity(
    entity: str,
    text: str,
    max_aliases: int = 5,
    *,
    strict: bool = False,
) -> list[str]:
    """Generate alias variants for an entity using normalization heuristics."""
    candidates = _generate_alias_candidates_with_confidence(entity, text)
    aliases, _review = _select_aliases_with_review(
        candidates,
        entity.strip(),
        strict=strict,
        max_aliases=max_aliases,
    )
    return aliases


def generate_metadata_from_entities(
    entities: list[str],
    text: str,
    *,
    enrichment: EnrichmentOptions | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate metadata entries from extracted entities."""
    if enrichment is None:
        enrichment = EnrichmentOptions()

    min_entity_length = 2
    generated_entries: list[dict[str, Any]] = []
    review_entries: list[dict[str, Any]] = []
    for entity in sorted(set(entities), key=str.lower):
        stripped_entity = entity.strip()
        if len(stripped_entity) < min_entity_length:
            continue

        entry: dict[str, Any] = {"uuid": str(uuid.uuid4()), "text": stripped_entity}
        review_item: dict[str, Any] = {
            "text": stripped_entity,
            "category": {"value": None, "confidence": None, "kept": False, "reason": "disabled"},
            "aliases": [],
        }

        if enrichment.auto_categories:
            category, confidence = infer_category_with_confidence(stripped_entity, text)
            category_kept = not enrichment.strict or confidence >= CATEGORY_STRICT_THRESHOLD
            if category_kept:
                entry["category"] = category
            review_item["category"] = {
                "value": category,
                "confidence": round(confidence, 3),
                "kept": category_kept,
                "reason": "kept" if category_kept else "strict_low_confidence",
            }

        if enrichment.auto_aliases:
            alias_candidates = _generate_alias_candidates_with_confidence(stripped_entity, text)
            aliases, alias_review = _select_aliases_with_review(
                alias_candidates,
                stripped_entity,
                strict=enrichment.strict,
                max_aliases=enrichment.max_aliases,
            )
            if aliases:
                entry["aliases"] = aliases
            review_item["aliases"] = alias_review

        generated_entries.append(entry)
        review_entries.append(review_item)

    return generated_entries, review_entries


__all__ = [
    "_extract_context_windows",
    "_extract_parenthetical_aliases",
    "_generate_alias_candidates_with_confidence",
    "_select_aliases_with_review",
    "_split_camel_words",
    "extract_key_phrases",
    "extract_named_entities",
    "generate_aliases_for_entity",
    "generate_metadata_from_entities",
    "infer_category_for_entity",
    "infer_category_with_confidence",
]
