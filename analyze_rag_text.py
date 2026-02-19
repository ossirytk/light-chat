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
    potential_metadata: list[dict[str, str]]
    statistics: dict[str, Any]


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

    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]

    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)

    key_bigrams = [phrase for phrase, count in bigram_counts.items() if count >= min_freq]
    key_trigrams = [phrase for phrase, count in trigram_counts.items() if count >= min_freq]

    return key_trigrams[:20] + key_bigrams[:30]


def generate_metadata_from_entities(entities: list[str]) -> list[dict[str, str]]:
    """Generate metadata entries from extracted entities."""
    import uuid

    metadata = []
    for entity in entities:
        if len(entity) >= 2 and entity.strip():
            metadata.append({"uuid": str(uuid.uuid4()), "text": entity.strip()})

    return metadata


def analyze_text_file(file_path: Path, min_phrase_freq: int = 3) -> TextAnalysisResult:
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
    potential_metadata = generate_metadata_from_entities(named_entities[:100])

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
@click.option("--verbose", "-v", is_flag=True, help="Show detailed analysis results")
def analyze(file_path: Path, output: Path | None, min_freq: int, verbose: bool) -> None:
    """Analyze a text file and extract metadata and topics."""
    result = analyze_text_file(file_path, min_phrase_freq=min_freq)

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
            logger.error(f"Found {len(result['issues'])} issues:")
            for issue in result["issues"][:10]:
                logger.error(f"  - {issue}")

            if len(result["issues"]) > 10:
                logger.error(f"  ... and {len(result['issues']) - 10} more issues")


@cli.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--auto-generate", "-g", is_flag=True, help="Auto-generate missing metadata files")
def scan(directory: Path, auto_generate: bool) -> None:
    """Scan a directory for text files and their metadata."""
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
                result = analyze_text_file(txt_file)
                with json_file.open("w", encoding="utf-8") as f:
                    json.dump(result.potential_metadata, f, indent=2, ensure_ascii=False)
                logger.info(f"  ✓ Generated {len(result.potential_metadata)} entries")


if __name__ == "__main__":
    cli()
