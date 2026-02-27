"""Enhanced collection management script for ChromaDB.

This script provides comprehensive collection management:
- List all collections with statistics
- Delete single or multiple collections
- Test collections with similarity search
- Export collection data
- Collection backup and restore
- Bulk operations
"""

import fnmatch
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb
import click
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from core.collection_helper import extract_key_matches


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
class ManagementContext:
    client: chromadb.PersistentClient
    persist_directory: str
    embedder: HuggingFaceEmbeddings
    key_storage: str


def normalize_keyfile(raw_keys: object) -> list[dict[str, object]]:
    if isinstance(raw_keys, dict) and "Content" in raw_keys:
        raw_keys = raw_keys["Content"]
    if not isinstance(raw_keys, list):
        return []
    return [item for item in raw_keys if isinstance(item, dict)]


def build_where_filters(matches: list[dict[str, str]]) -> list[dict[str, object]]:
    if not matches:
        return [{}]
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


@click.group()
def cli() -> None:
    """Manage ChromaDB collections for RAG data."""
    app_config = load_app_config()
    configure_logging(app_config)


@cli.command()
@click.option(
    "--persist-directory",
    "-p",
    default=None,
    help="Directory where ChromaDB stores collections",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed collection information")
def list_collections(persist_directory: str | None, verbose: bool) -> None:
    """List all ChromaDB collections."""
    app_config = load_app_config()
    persist_directory = persist_directory or app_config.get("PERSIST_DIRECTORY", "./character_storage/")

    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))

    collections = client.list_collections()

    if not collections:
        logger.info("No collections found")
        return

    logger.info(f"Found {len(collections)} collection(s):")

    for collection in collections:
        if verbose:
            info = get_collection_info(client, collection.name)
            logger.info(f"  • {info['name']}")
            logger.info(f"    - Documents: {info['count']}")
            logger.info(f"    - Metadata: {info['metadata']}")
        else:
            logger.info(f"  • {collection.name} ({collection.count()} documents)")


@cli.command()
@click.argument("collection_name")
@click.option(
    "--persist-directory",
    "-p",
    default=None,
    help="Directory where ChromaDB stores collections",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def delete(collection_name: str, persist_directory: str | None, yes: bool) -> None:
    """Delete a ChromaDB collection."""
    app_config = load_app_config()
    persist_directory = persist_directory or app_config.get("PERSIST_DIRECTORY", "./character_storage/")

    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))

    info = get_collection_info(client, collection_name)

    if not info["exists"]:
        logger.error(f"Collection '{collection_name}' not found")
        return

    if not yes:
        logger.warning(f"About to delete collection: {collection_name}")
        logger.warning(f"This collection contains {info['count']} documents")
        confirmation = input("Are you sure? (yes/no): ")
        if confirmation.lower() not in ["yes", "y"]:
            logger.info("Deletion cancelled")
            return

    try:
        client.delete_collection(collection_name)
        logger.info(f"✓ Deleted collection: {collection_name}")
    except ValueError as e:
        logger.error(f"Error deleting collection: {e}")


@cli.command()
@click.option(
    "--persist-directory",
    "-p",
    default=None,
    help="Directory where ChromaDB stores collections",
)
@click.option("--pattern", default=None, help="Delete collections matching pattern (use * for wildcard)")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def delete_multiple(persist_directory: str | None, pattern: str | None, yes: bool) -> None:
    """Delete multiple collections matching a pattern."""
    app_config = load_app_config()
    persist_directory = persist_directory or app_config.get("PERSIST_DIRECTORY", "./character_storage/")

    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))

    collections = client.list_collections()

    if not pattern:
        logger.error("Must specify --pattern for bulk deletion")
        return

    matching = [c for c in collections if fnmatch.fnmatch(c.name, pattern)]

    if not matching:
        logger.info(f"No collections match pattern: {pattern}")
        return

    logger.info(f"Found {len(matching)} collection(s) matching pattern '{pattern}':")
    for collection in matching:
        logger.info(f"  • {collection.name} ({collection.count()} documents)")

    if not yes:
        confirmation = input(f"Delete all {len(matching)} collections? (yes/no): ")
        if confirmation.lower() not in ["yes", "y"]:
            logger.info("Deletion cancelled")
            return

    for collection in matching:
        try:
            client.delete_collection(collection.name)
            logger.info(f"✓ Deleted: {collection.name}")
        except ValueError as e:
            logger.error(f"✗ Error deleting {collection.name}: {e}")


@cli.command()
@click.argument("collection_name")
@click.option("--query", "-q", required=True, help="Query text to search")
@click.option("--k", default=5, type=int, help="Number of results to return")
@click.option(
    "--persist-directory",
    "-p",
    default=None,
    help="Directory where ChromaDB stores collections",
)
@click.option(
    "--key-storage",
    "-k",
    default=None,
    help="Directory containing metadata JSON files",
)
def test(
    collection_name: str,
    query: str,
    k: int,
    persist_directory: str | None,
    key_storage: str | None,
) -> None:
    """Test a collection with a similarity search query."""
    app_config = load_app_config()
    persist_directory = persist_directory or app_config.get("PERSIST_DIRECTORY", "./character_storage/")
    key_storage = key_storage or app_config.get("KEY_STORAGE", "./rag_data/")

    embedding_device = str(app_config.get("EMBEDDING_DEVICE", "cpu"))
    embedding_cache = str(app_config.get("EMBEDDING_CACHE", "./embedding_models/"))
    model_kwargs = {"device": embedding_device}
    encode_kwargs = {"normalize_embeddings": False}
    cache_folder = str(Path(embedding_cache))

    embedder = HuggingFaceEmbeddings(
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder=cache_folder,
    )

    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))

    info = get_collection_info(client, collection_name)
    if not info["exists"]:
        logger.error(f"Collection '{collection_name}' not found")
        return

    logger.info(f"Testing collection: {collection_name}")
    logger.info(f"Query: {query}")
    logger.info(f"Total documents: {info['count']}")

    base_name = collection_name.replace("_mes", "")
    keyfile_path = Path(key_storage) / f"{base_name}.json"

    filters = [{}]
    if keyfile_path.exists():
        logger.info(f"Loading metadata from: {keyfile_path}")
        with keyfile_path.open(encoding="utf-8") as f:
            key_data = json.load(f)
        keys = normalize_keyfile(key_data)
        matches = extract_key_matches(keys, query)
        if matches:
            logger.info(f"Matched {len(matches)} metadata key(s) from query")
            filters = build_where_filters(matches)

    db = Chroma(
        client=client,
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedder,
    )

    for filter_idx, where_filter in enumerate(filters):
        logger.info(f"\nAttempting search with filter #{filter_idx + 1}")
        docs = db.similarity_search_with_score(query=query, k=k, filter=where_filter)

        if docs:
            logger.info(f"✓ Found {len(docs)} result(s)")
            for idx, (doc, score) in enumerate(docs, 1):
                logger.info(f"\nResult {idx} (score: {score:.4f}):")
                logger.info(f"Content preview: {doc.page_content[:200]}...")
                if doc.metadata:
                    logger.info(f"Metadata keys: {list(doc.metadata.keys())[:5]}")
            return

    logger.info("No results found")


@cli.command()
@click.argument("collection_name")
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Output JSON file")
@click.option(
    "--persist-directory",
    "-p",
    default=None,
    help="Directory where ChromaDB stores collections",
)
def export(collection_name: str, output: Path, persist_directory: str | None) -> None:
    """Export collection data to JSON."""
    app_config = load_app_config()
    persist_directory = persist_directory or app_config.get("PERSIST_DIRECTORY", "./character_storage/")

    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))

    info = get_collection_info(client, collection_name)
    if not info["exists"]:
        logger.error(f"Collection '{collection_name}' not found")
        return

    logger.info(f"Exporting collection: {collection_name}")

    collection = client.get_collection(collection_name)
    data = collection.get()

    export_data = {
        "collection_name": collection_name,
        "count": len(data.get("ids", [])),
        "metadata": collection.metadata,
        "documents": data.get("documents", []),
        "metadatas": data.get("metadatas", []),
        "ids": data.get("ids", []),
    }

    with output.open("w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Exported {export_data['count']} documents to {output}")


@cli.command()
@click.argument("collection_name")
@click.option(
    "--persist-directory",
    "-p",
    default=None,
    help="Directory where ChromaDB stores collections",
)
def info(collection_name: str, persist_directory: str | None) -> None:
    """Show detailed information about a collection."""
    app_config = load_app_config()
    persist_directory = persist_directory or app_config.get("PERSIST_DIRECTORY", "./character_storage/")

    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))

    info = get_collection_info(client, collection_name)

    if not info["exists"]:
        logger.error(f"Collection '{collection_name}' not found")
        return

    collection = client.get_collection(collection_name)
    sample = collection.peek(limit=1)

    logger.info(f"Collection: {collection_name}")
    logger.info(f"Documents: {info['count']}")
    logger.info(f"Metadata: {info['metadata']}")

    if sample.get("documents"):
        logger.info("\nSample document:")
        logger.info(f"  Content: {sample['documents'][0][:150]}...")
        if sample.get("metadatas") and sample["metadatas"][0]:
            logger.info(f"  Metadata keys: {list(sample['metadatas'][0].keys())}")


if __name__ == "__main__":
    cli()
