"""Collection and embedding helpers for collection management core."""

import chromadb
import click
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import load_app_config, load_rag_script_config
from scripts.rag.manage_collections_core_types import (
    EMBEDDING_DIMENSION_METADATA_KEY,
    EMBEDDING_MODEL_METADATA_KEY,
    EMBEDDING_NORMALIZE_METADATA_KEY,
    KeyItem,
    KeyMatch,
    WhereFilter,
)

MISSING_COLLECTION_ERRORS = (ValueError, chromadb.errors.NotFoundError)


def infer_embedding_dimension(embedder: HuggingFaceEmbeddings) -> int | None:
    try:
        vector = embedder.embed_query("dimension_probe")
    except Exception:
        return None
    return len(vector) if isinstance(vector, list) else None


def build_embedding_fingerprint(
    embedding_model: str,
    normalize_embeddings: bool,
    embedding_dimension: int | None,
) -> dict[str, object]:
    fingerprint: dict[str, object] = {
        EMBEDDING_MODEL_METADATA_KEY: embedding_model,
        EMBEDDING_NORMALIZE_METADATA_KEY: normalize_embeddings,
    }
    if embedding_dimension is not None:
        fingerprint[EMBEDDING_DIMENSION_METADATA_KEY] = embedding_dimension
    return fingerprint


def assert_collection_fingerprint_compatible(
    client: chromadb.PersistentClient,
    collection_name: str,
    expected_fingerprint: dict[str, object],
) -> None:
    try:
        collection = client.get_collection(collection_name)
    except MISSING_COLLECTION_ERRORS:
        return

    metadata = collection.metadata or {}
    mismatches = [
        (key, metadata[key], expected_fingerprint[key])
        for key in expected_fingerprint
        if key in metadata and metadata[key] != expected_fingerprint[key]
    ]

    if mismatches:
        mismatch_summary = ", ".join(
            f"{key}: existing={actual!r} expected={expected!r}" for key, actual, expected in mismatches
        )
        msg = (
            f"Collection '{collection_name}' has incompatible embedding fingerprint; refusing mixed-model read. "
            f"{mismatch_summary}"
        )
        raise click.ClickException(msg)


def _resolve_embedding_runtime(
    embedding_model: str | None,
    embedding_device: str | None,
) -> tuple[str, str, str]:
    app_config = load_app_config()
    script_config = load_rag_script_config(app_config)
    resolved_model = embedding_model or script_config.embedding_model
    resolved_device = embedding_device or script_config.embedding_device
    return resolved_model, resolved_device, script_config.embedding_cache


def normalize_keyfile(raw_keys: object) -> list[KeyItem]:
    if isinstance(raw_keys, dict) and "Content" in raw_keys:
        raw_keys = raw_keys["Content"]
    if not isinstance(raw_keys, list):
        return []
    return [item for item in raw_keys if isinstance(item, dict)]


def _get_entry_value(item: KeyItem) -> str | None:
    text_keys = ("text", "text_fields", "text_field", "content", "value")
    for key in text_keys:
        candidate = item.get(key)
        if isinstance(candidate, str):
            return candidate
    for key, candidate in item.items():
        if key in ("uuid", "aliases", "category"):
            continue
        if isinstance(candidate, str):
            return candidate
    return None


def _matches_aliases(item: KeyItem, text_lower: str) -> bool:
    aliases = item.get("aliases")
    if not isinstance(aliases, list):
        return False
    return any(isinstance(alias, str) and alias.lower() in text_lower for alias in aliases)


def extract_key_matches(keys: list[KeyItem], text: str) -> list[KeyMatch]:
    if not text:
        return []
    text_lower = text.lower()
    matches: list[KeyMatch] = []
    for item in keys:
        uuid = item.get("uuid")
        if not isinstance(uuid, str):
            continue
        value = _get_entry_value(item)
        if not isinstance(value, str):
            continue
        if value.lower() in text_lower or _matches_aliases(item, text_lower):
            matches.append({uuid: value})
    return matches


def build_where_filters(matches: list[KeyMatch]) -> list[WhereFilter]:
    if not matches:
        return [None]
    if len(matches) == 1:
        return [matches[0]]
    return [{"$and": matches}, {"$or": matches}]


def get_collection_info(client: chromadb.PersistentClient, collection_name: str) -> dict:
    """Get detailed information about a collection."""
    try:
        collection = client.get_collection(collection_name)
        count = collection.count()
        metadata = collection.metadata

    except ValueError:
        return {
            "name": collection_name,
            "exists": False,
        }
    else:
        return {
            "name": collection_name,
            "count": count,
            "metadata": metadata,
            "exists": True,
        }


__all__ = [
    "_get_entry_value",
    "_matches_aliases",
    "_resolve_embedding_runtime",
    "assert_collection_fingerprint_compatible",
    "build_embedding_fingerprint",
    "build_where_filters",
    "extract_key_matches",
    "get_collection_info",
    "infer_embedding_dimension",
    "normalize_keyfile",
]
