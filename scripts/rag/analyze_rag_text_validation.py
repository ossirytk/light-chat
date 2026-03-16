"""Metadata validation helpers for RAG text analysis outputs."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger


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
    duplicate_uuids = [entry_uuid for entry_uuid, count in Counter(uuid_values).items() if count > 1]

    if duplicate_uuids:
        issues.append(f"Duplicate UUIDs found: {duplicate_uuids[:5]}")

    return {
        "valid": len(issues) == 0,
        "total_entries": len(data),
        "issues": issues,
        "duplicate_uuid_count": len(duplicate_uuids),
    }


__all__ = ["validate_metadata_file"]
