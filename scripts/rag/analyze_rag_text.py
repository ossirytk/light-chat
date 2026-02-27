"""Analyze RAG text files to extract metadata and topics.

This script provides comprehensive text analysis capabilities for RAG data:
- Extract potential metadata keys from text
- Identify topics and themes using statistical analysis
- Generate metadata JSON files from text content
- Validate existing metadata files
"""

import json
import logging
import re
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
from loguru import logger


def load_app_config() -> dict:
    config_path = Path("./configs/") / "appconf.json"
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        return json.load(f)


def configure_logging(app_config: dict) -> None:
    show_logs = bool(app_config.get("SHOW_LOGS", True))
    log_level = str(app_config.get("LOG_LEVEL", "DEBUG")).upper()
    if show_logs:
        logging.basicConfig(level=log_level)
        logger.remove()
        logger.add(sys.stderr, level=log_level)
    else:
        logging.disable(logging.CRITICAL)
        logger.remove()


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


def extract_named_entities(text: str) -> list[str]:
    """Extract potential named entities using heuristics.

    This uses simple pattern matching to identify:
    - Capitalized words/phrases
    - Quoted text
    - Numbers and dates
    """
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


def analyze_text_file(
    file_path: Path,
    min_phrase_freq: int = 3,
    *,
    enrichment: EnrichmentOptions | None = None,
) -> TextAnalysisResult:
    """Analyze a text file and extract metadata and topics."""
    logger.info(f"Analyzing file: {file_path}")

    if not file_path.exists():
        msg = f"File not found: {file_path}"
        raise FileNotFoundError(msg)

    with file_path.open(encoding="utf-8") as f:
        text = f.read()

    lines = text.split("\n")
    words = re.findall(r"\b\w+\b", text.lower())

    named_entities = extract_named_entities(text)
    key_phrases = extract_key_phrases(text, min_freq=min_phrase_freq)
    potential_metadata, enrichment_review = generate_metadata_from_entities(
        named_entities[:100],
        text,
        enrichment=enrichment,
    )

    word_counts = Counter(words)
    common_words = word_counts.most_common(20)

    statistics = {
        "common_words": [{"word": word, "count": count} for word, count in common_words],
        "entity_count": len(named_entities),
        "key_phrase_count": len(key_phrases),
    }

    return TextAnalysisResult(
        total_chars=len(text),
        total_lines=len(lines),
        total_words=len(words),
        unique_words=len(set(words)),
        named_entities=named_entities[:50],
        key_phrases=key_phrases,
        potential_metadata=potential_metadata,
        enrichment_review=enrichment_review,
        statistics=statistics,
    )


def validate_metadata_file(metadata_path: Path) -> dict[str, Any]:
    """Validate a metadata JSON file structure."""
    logger.info(f"Validating metadata file: {metadata_path}")

    if not metadata_path.exists():
        return {"valid": False, "error": "File not found", "issues": []}

    try:
        with metadata_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return {"valid": False, "error": f"Invalid JSON: {e}", "issues": []}

    issues = []

    if isinstance(data, dict) and "Content" in data:
        data = data["Content"]

    if not isinstance(data, list):
        return {"valid": False, "error": "Data must be a list or dict with 'Content' key", "issues": []}

    text_keys = ("text", "text_fields", "text_field", "content", "value")

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            issues.append(f"Item {idx}: Not a dictionary")
            continue

        if "uuid" not in item:
            issues.append(f"Item {idx}: Missing 'uuid' field")

        has_text_field = any(key in item for key in text_keys)
        if not has_text_field:
            other_text_keys = [k for k, v in item.items() if k != "uuid" and isinstance(v, str)]
            if not other_text_keys:
                issues.append(f"Item {idx}: Missing text field (expected one of {text_keys})")

    uuid_values = [item.get("uuid") for item in data if isinstance(item, dict) and "uuid" in item]
    duplicate_uuids = [uuid for uuid, count in Counter(uuid_values).items() if count > 1]

    if duplicate_uuids:
        issues.append(f"Duplicate UUIDs found: {duplicate_uuids[:5]}")

    return {
        "valid": len(issues) == 0,
        "total_entries": len(data),
        "issues": issues,
        "duplicate_uuid_count": len(duplicate_uuids),
    }


@click.group()
def cli() -> None:
    """Analyze RAG text files and extract metadata."""
    app_config = load_app_config()
    configure_logging(app_config)


@cli.command()
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output JSON file for extracted metadata")
@click.option("--min-freq", "-f", default=3, type=int, help="Minimum frequency for key phrase extraction")
@click.option(
    "--auto-categories/--no-auto-categories",
    default=True,
    help="Enable or disable metadata category field generation",
)
@click.option(
    "--auto-aliases/--no-auto-aliases",
    default=True,
    help="Enable or disable metadata alias variant generation",
)
@click.option("--max-aliases", default=5, type=int, help="Maximum alias count to generate per metadata entry")
@click.option("--strict", is_flag=True, help="Emit only high-confidence category and alias enrichments")
@click.option(
    "--review-report",
    type=click.Path(path_type=Path),
    help="Output JSON file with enrichment decision details",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed analysis results")
def analyze(  # noqa: PLR0913
    file_path: Path,
    output: Path | None,
    min_freq: int,
    auto_categories: bool,
    auto_aliases: bool,
    max_aliases: int,
    strict: bool,
    review_report: Path | None,
    verbose: bool,
) -> None:
    """Analyze a text file and extract metadata and topics."""
    enrichment = EnrichmentOptions(
        auto_categories=auto_categories,
        auto_aliases=auto_aliases,
        max_aliases=max_aliases,
        strict=strict,
    )

    result = analyze_text_file(
        file_path,
        min_phrase_freq=min_freq,
        enrichment=enrichment,
    )

    logger.info(f"Analysis complete for {file_path.name}")
    logger.info(f"Total characters: {result.total_chars}")
    logger.info(f"Total lines: {result.total_lines}")
    logger.info(f"Total words: {result.total_words}")
    logger.info(f"Unique words: {result.unique_words}")
    logger.info(f"Named entities found: {result.statistics['entity_count']}")
    logger.info(f"Key phrases found: {result.statistics['key_phrase_count']}")

    if verbose:
        logger.info("\nTop named entities (sample):")
        for entity in result.named_entities[:20]:
            logger.info(f"  - {entity}")

        logger.info("\nKey phrases (sample):")
        for phrase in result.key_phrases[:10]:
            logger.info(f"  - {phrase}")

        logger.info("\nMost common words:")
        for word_info in result.statistics["common_words"][:10]:
            logger.info(f"  - {word_info['word']}: {word_info['count']}")

    if output:
        with output.open("w", encoding="utf-8") as f:
            json.dump(result.potential_metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Metadata saved to: {output}")
        logger.info(f"Generated {len(result.potential_metadata)} metadata entries")

    if review_report:
        with review_report.open("w", encoding="utf-8") as f:
            json.dump(result.enrichment_review, f, indent=2, ensure_ascii=False)
        logger.info(f"Enrichment review report saved to: {review_report}")


@cli.command()
@click.argument("metadata_path", type=click.Path(exists=True, path_type=Path))
def validate(metadata_path: Path) -> None:
    """Validate a metadata JSON file."""
    result = validate_metadata_file(metadata_path)

    if result["valid"]:
        logger.info(f"✓ Metadata file is valid: {metadata_path}")
        logger.info(f"Total entries: {result['total_entries']}")
    else:
        logger.error(f"✗ Metadata file has issues: {metadata_path}")
        if "error" in result:
            logger.error(f"Error: {result['error']}")

        if result.get("issues"):
            max_issues_to_display = 10
            logger.error(f"Found {len(result['issues'])} issues:")
            for issue in result["issues"][:max_issues_to_display]:
                logger.error(f"  - {issue}")

            if len(result["issues"]) > max_issues_to_display:
                logger.error(f"  ... and {len(result['issues']) - max_issues_to_display} more issues")


@cli.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--auto-generate", "-g", is_flag=True, help="Auto-generate missing metadata files")
@click.option(
    "--auto-categories/--no-auto-categories",
    default=True,
    help="Enable or disable metadata category field generation",
)
@click.option(
    "--auto-aliases/--no-auto-aliases",
    default=True,
    help="Enable or disable metadata alias variant generation",
)
@click.option("--max-aliases", default=5, type=int, help="Maximum alias count to generate per metadata entry")
@click.option("--strict", is_flag=True, help="Emit only high-confidence category and alias enrichments")
def scan(  # noqa: PLR0913
    directory: Path,
    auto_generate: bool,
    auto_categories: bool,
    auto_aliases: bool,
    max_aliases: int,
    strict: bool,
) -> None:
    """Scan a directory for text files and their metadata."""
    enrichment = EnrichmentOptions(
        auto_categories=auto_categories,
        auto_aliases=auto_aliases,
        max_aliases=max_aliases,
        strict=strict,
    )

    txt_files = list(directory.glob("*.txt"))
    json_files = list(directory.glob("*.json"))

    logger.info(f"Scanning directory: {directory}")
    logger.info(f"Found {len(txt_files)} text files")
    logger.info(f"Found {len(json_files)} JSON files")

    for txt_file in txt_files:
        base_name = txt_file.stem
        if base_name.endswith("_message_examples"):
            base_name = base_name.replace("_message_examples", "")

        json_file = directory / f"{base_name}.json"

        if json_file.exists():
            logger.info(f"✓ {txt_file.name} has metadata: {json_file.name}")
            validation = validate_metadata_file(json_file)
            if not validation["valid"]:
                logger.warning(f"  ⚠ Metadata has {len(validation.get('issues', []))} issues")
        else:
            logger.warning(f"✗ {txt_file.name} missing metadata: {json_file.name}")

            if auto_generate:
                logger.info(f"  → Generating metadata for {txt_file.name}...")
                result = analyze_text_file(
                    txt_file,
                    enrichment=enrichment,
                )
                with json_file.open("w", encoding="utf-8") as f:
                    json.dump(result.potential_metadata, f, indent=2, ensure_ascii=False)
                logger.info(f"  ✓ Generated {len(result.potential_metadata)} entries")


if __name__ == "__main__":
    cli()
