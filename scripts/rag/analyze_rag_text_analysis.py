"""File-level text analysis orchestration for RAG metadata workflows."""

import re
from collections import Counter
from pathlib import Path

from loguru import logger

from scripts.rag.analyze_rag_text_enrichment import (
    extract_key_phrases,
    extract_named_entities,
    generate_metadata_from_entities,
)
from scripts.rag.analyze_rag_text_types import EnrichmentOptions, TextAnalysisResult


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

    with file_path.open(encoding="utf-8") as file_handle:
        text = file_handle.read()

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


__all__ = ["analyze_text_file"]
