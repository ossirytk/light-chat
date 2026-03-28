"""Metadata validation helpers for RAG text analysis outputs."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger

_TEXT_KEYS = ("text", "text_fields", "text_field", "content", "value")


def _validate_items(data: list[Any]) -> list[str]:
    """Return a list of per-item validation issue messages."""
    issues: list[str] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            issues.append(f"Item {idx}: Not a dictionary")
            continue

        if "uuid" not in item:
            issues.append(f"Item {idx}: Missing 'uuid' field")

        if not any(key in item for key in _TEXT_KEYS):
            other_text_keys = [k for k, v in item.items() if k != "uuid" and isinstance(v, str)]
            if not other_text_keys:
                issues.append(f"Item {idx}: Missing text field (expected one of {_TEXT_KEYS})")

    return issues


def _check_duplicate_uuids(data: list[Any]) -> list[Any]:
    """Return list of UUID values that appear more than once."""
    uuid_values = [item.get("uuid") for item in data if isinstance(item, dict) and "uuid" in item]
    return [u for u, count in Counter(uuid_values).items() if count > 1]


def validate_metadata_file(metadata_path: Path) -> dict[str, Any]:
    """Validate a metadata JSON file structure."""
    logger.info(f"Validating metadata file: {metadata_path}")

    if not metadata_path.exists():
        return {"valid": False, "error": "File not found", "issues": []}

    try:
        with metadata_path.open(encoding="utf-8") as file_handle:
            data = json.load(file_handle)
    except json.JSONDecodeError as error:
        return {"valid": False, "error": f"Invalid JSON: {error}", "issues": []}

    if isinstance(data, dict) and "Content" in data:
        data = data["Content"]

    if not isinstance(data, list):
        return {"valid": False, "error": "Data must be a list or dict with 'Content' key", "issues": []}

    issues = _validate_items(data)
    duplicate_uuids = _check_duplicate_uuids(data)

    if duplicate_uuids:
        issues.append(f"Duplicate UUIDs found: {duplicate_uuids[:5]}")

    return {
        "valid": len(issues) == 0,
        "total_entries": len(data),
        "issues": issues,
        "duplicate_uuid_count": len(duplicate_uuids),
    }


__all__ = ["validate_metadata_file"]
