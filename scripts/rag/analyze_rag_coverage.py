"""Source document coverage scoring for RAG metadata quality assessment.

Computes what fraction of source text is represented in metadata entities,
helping identify incomplete metadata before pushing to ChromaDB.
"""

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# Constants for magic numbers
MIN_TEXT_LENGTH = 3
MIN_SEGMENT_LENGTH = 20
MAX_DISPLAY_LENGTH = 100
REPORT_DISPLAY_TRUNCATE = 60


@dataclass
class CoverageMetrics:
    """Coverage metrics for a source document."""

    entities_count: int
    source_coverage_ratio: float
    total_source_chars: int
    covered_chars: int
    unmapped_segments: list[str]
    category_distribution: dict[str, int]
    entity_coverage: dict[str, dict[str, Any]]


def _fuzzy_match(text: str, source: str, threshold: float = 0.8) -> bool:
    """Check if text appears in source with fuzzy matching."""
    if not text or len(text) < MIN_TEXT_LENGTH:
        return False

    text_lower = text.lower()
    source_lower = source.lower()

    if text_lower in source_lower:
        return True

    for line in source_lower.split("\n"):
        ratio = SequenceMatcher(None, text_lower, line).ratio()
        if ratio >= threshold:
            return True

    return False


def _extract_matched_segments(  # noqa: PLR0912
    source_text: str,
    metadata_texts: list[str],
) -> tuple[int, list[str]]:
    """Identify which parts of source text are covered by metadata.

    Returns:
        Tuple of (covered_char_count, unmapped_segment_list)
    """
    source_lower = source_text.lower()
    marked = [False] * len(source_text)

    for meta_text in metadata_texts:
        if not meta_text:
            continue
        meta_lower = meta_text.lower()
        start_pos = 0
        while True:
            pos = source_lower.find(meta_lower, start_pos)
            if pos == -1:
                break
            for i in range(pos, pos + len(meta_text)):
                if i < len(marked):
                    marked[i] = True
            start_pos = pos + 1

    unmapped_segments = []
    current_segment = []
    for _i, (char, is_covered) in enumerate(zip(source_text, marked, strict=True)):
        if not is_covered:
            current_segment.append(char)
        else:
            if current_segment and "".join(current_segment).strip():
                seg_text = "".join(current_segment).strip()
                if len(seg_text) > MIN_SEGMENT_LENGTH:
                    truncated = (
                        seg_text[:MAX_DISPLAY_LENGTH] + "..." if len(seg_text) > MAX_DISPLAY_LENGTH else seg_text
                    )
                    unmapped_segments.append(truncated)
            current_segment = []

    if current_segment and "".join(current_segment).strip():
        seg_text = "".join(current_segment).strip()
        if len(seg_text) > MIN_SEGMENT_LENGTH:
            truncated = seg_text[:MAX_DISPLAY_LENGTH] + "..." if len(seg_text) > MAX_DISPLAY_LENGTH else seg_text
            unmapped_segments.append(truncated)

    covered_chars = sum(1 for c in marked if c)
    return covered_chars, unmapped_segments


def extract_coverage_metrics(
    source_text: str,
    metadata_list: list[dict[str, Any]],
) -> CoverageMetrics:
    """Compute coverage metrics for a source document against metadata.

    Args:
        source_text: Full source document text
        metadata_list: List of metadata items (each with 'uuid', 'text', 'category' keys)

    Returns:
        CoverageMetrics object with detailed breakdown
    """
    if not metadata_list:
        return CoverageMetrics(
            entities_count=0,
            source_coverage_ratio=0.0,
            total_source_chars=len(source_text),
            covered_chars=0,
            unmapped_segments=[],
            category_distribution={},
            entity_coverage={},
        )

    entities = [item for item in metadata_list if isinstance(item, dict) and "uuid" in item]
    entity_texts = []
    category_dist: dict[str, int] = {}
    entity_cov: dict[str, dict[str, Any]] = {}

    for entity in entities:
        entity_text = entity.get("text", "")
        if entity_text:
            entity_texts.append(entity_text)
            found = _fuzzy_match(entity_text, source_text)
            category = entity.get("category", "unknown")
            if category:
                category_dist[category] = category_dist.get(category, 0) + 1
            entity_cov[entity.get("uuid", "unknown")] = {
                "text": entity_text,
                "category": category,
                "found": found,
            }

    covered_chars, unmapped_segs = _extract_matched_segments(source_text, entity_texts)
    total_source_chars = len(source_text)
    coverage_ratio = covered_chars / total_source_chars if total_source_chars > 0 else 0.0

    return CoverageMetrics(
        entities_count=len(entities),
        source_coverage_ratio=coverage_ratio,
        total_source_chars=total_source_chars,
        covered_chars=covered_chars,
        unmapped_segments=unmapped_segs[:10],
        category_distribution=category_dist,
        entity_coverage=entity_cov,
    )


def load_metadata_file(metadata_path: Path) -> list[dict[str, Any]]:
    """Load and normalize metadata from JSON file."""
    if not metadata_path.exists():
        msg = f"Metadata file not found: {metadata_path}"
        raise FileNotFoundError(msg)

    with metadata_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "Content" in data:
        data = data["Content"]

    if not isinstance(data, list):
        msg = f"Expected metadata to be a list, got {type(data)}"
        raise TypeError(msg)

    return data


def format_coverage_report(metrics: CoverageMetrics, threshold: float = 0.75) -> str:
    """Format coverage metrics as a human-readable report."""
    lines = [
        "=" * 70,
        "RAG SOURCE COVERAGE ANALYSIS",
        "=" * 70,
        f"Entities found: {metrics.entities_count}",
        f"Source chars total: {metrics.total_source_chars}",
        f"Source chars covered: {metrics.covered_chars}",
        f"Coverage ratio: {metrics.source_coverage_ratio * 100:.1f}%",
        (
            f"Status: {'✓ PASS' if metrics.source_coverage_ratio >= threshold else '✗ FAIL'}"
            f" (threshold: {threshold * 100:.0f}%)"
        ),
        "",
    ]

    if metrics.category_distribution:
        lines.append("Category breakdown:")
        for cat, count in sorted(metrics.category_distribution.items()):
            lines.append(f"  {cat:15} {count:3} items")
        lines.append("")

    if metrics.unmapped_segments:
        lines.append(f"Top {len(metrics.unmapped_segments)} unmapped text segments:")
        for seg in metrics.unmapped_segments[:5]:
            truncated = seg[:REPORT_DISPLAY_TRUNCATE] + "..." if len(seg) > REPORT_DISPLAY_TRUNCATE else seg
            lines.append(f"  - {truncated}")
        lines.append("")

    not_found = [item for item in metrics.entity_coverage.values() if not item["found"]]
    if not_found:
        lines.append(f"Entities not found in source ({len(not_found)}/{len(metrics.entity_coverage)}):")
        lines.extend(f"  - {item['text'][:50]}" for item in not_found[:10])
        lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)
