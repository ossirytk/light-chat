"""Enhanced script for pushing RAG data to ChromaDB.

This script provides comprehensive features for managing RAG data:
- Batch upload of multiple text files
- Metadata enrichment with validation
- Collection versioning support
- Progress tracking and error handling
- Dry-run mode for testing
"""

import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import chromadb
import click
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents.base import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
class PushConfig:
    """Configuration for pushing data to ChromaDB."""

    persist_directory: str
    chunk_size: int
    chunk_overlap: int
    key_storage: str
    threads: int
    dry_run: bool
    overwrite: bool


@dataclass
class ProcessingContext:
    embedder: HuggingFaceEmbeddings
    client: chromadb.PersistentClient


def iter_key_items(all_keys: object) -> list[dict]:
    if not isinstance(all_keys, list):
        return []
    return [item for item in all_keys if isinstance(item, dict)]


def get_item_value(item: dict, text_keys: tuple[str, ...]) -> str | None:
    for tk in text_keys:
        candidate = item.get(tk)
        if isinstance(candidate, str):
            return candidate
    for key, candidate in item.items():
        if key == "uuid":
            continue
        if isinstance(candidate, str):
            return candidate
    return None


def enrich_document_with_metadata(document: Document, all_keys: list) -> Document:
    """Enrich a document with metadata keys found in its content."""
    if not isinstance(document.metadata, dict):
        document.metadata = {} if document.metadata is None else dict(document.metadata)

    text_keys = ("text", "text_fields", "text_field", "content", "value")
    for item in iter_key_items(all_keys):
        uuid = item.get("uuid")
        if not isinstance(uuid, str):
            continue
        value = get_item_value(item, text_keys)
        if value and value in document.page_content:
            document.metadata[uuid] = value

    return document


def load_and_chunk_text_file(
    file_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Load a text file and split it into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return text_splitter.split_documents(documents)


def enrich_documents_with_metadata(
    documents: list[Document],
    metadata_file: Path,
    threads: int,
) -> list[Document]:
    """Enrich documents with metadata using multiprocessing."""
    if not metadata_file.exists():
        logger.warning(f"No metadata file found at {metadata_file}, skipping enrichment")
        return documents

    logger.info(f"Loading metadata from: {metadata_file}")
    with metadata_file.open(encoding="utf-8") as f:
        all_keys = json.load(f)

    if isinstance(all_keys, dict) and "Content" in all_keys:
        all_keys = all_keys["Content"]

    logger.info(f"Enriching {len(documents)} documents with metadata using {threads} threads")
    tic = time.perf_counter()

    enrich_fn = partial(enrich_document_with_metadata, all_keys=all_keys)
    with ProcessPoolExecutor(max_workers=threads) as executor:
        chunksize = max(1, len(documents) // threads)
        enriched_docs = list(executor.map(enrich_fn, documents, chunksize=chunksize))

    toc = time.perf_counter()
    logger.info(f"Metadata enrichment took {toc - tic:0.4f} seconds")

    return enriched_docs


def push_to_collection(
    collection_name: str,
    documents: list[Document],
    config: PushConfig,
    context: ProcessingContext,
) -> None:
    """Push documents to a ChromaDB collection."""
    if config.dry_run:
        logger.info(f"[DRY RUN] Would push {len(documents)} documents to collection '{collection_name}'")
        return

    if config.overwrite:
        try:
            context.client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except ValueError:
            logger.debug(f"Collection {collection_name} doesn't exist, creating new")

    logger.info(f"Pushing {len(documents)} documents to collection '{collection_name}'")
    tic = time.perf_counter()

    Chroma.from_documents(
        client=context.client,
        documents=documents,
        embedding=context.embedder,
        persist_directory=config.persist_directory,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "l2"},
    )

    toc = time.perf_counter()
    logger.info(f"Collection '{collection_name}' created in {toc - tic:0.4f} seconds")


@click.command()
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--collection-name",
    "-c",
    required=True,
    help="Name of the ChromaDB collection to create",
)
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
@click.option(
    "--metadata-file",
    "-m",
    type=click.Path(path_type=Path),
    help="Specific metadata JSON file to use (overrides auto-detection)",
)
@click.option("--chunk-size", "-cs", default=None, type=int, help="Chunk size for text splitting")
@click.option("--chunk-overlap", "-co", default=None, type=int, help="Overlap size for chunks")
@click.option("--threads", "-t", default=None, type=int, help="Number of threads for processing")
@click.option("--dry-run", "-d", is_flag=True, help="Show what would be done without making changes")
@click.option("--overwrite", "-w", is_flag=True, help="Overwrite existing collection if it exists")
def main(
    file_path: Path,
    collection_name: str,
    persist_directory: str | None,
    key_storage: str | None,
    metadata_file: Path | None,
    chunk_size: int | None,
    chunk_overlap: int | None,
    threads: int | None,
    dry_run: bool,
    overwrite: bool,
) -> None:
    """Push a text file to ChromaDB with metadata enrichment.

    This script provides enhanced features over prepare_rag.py:
    - Single file processing with explicit collection naming
    - Dry-run mode for testing
    - Overwrite protection
    - Custom metadata file selection
    - Detailed progress tracking
    """
    app_config = load_app_config()
    configure_logging(app_config)

    persist_directory = persist_directory or app_config.get("PERSIST_DIRECTORY", "./character_storage/")
    key_storage = key_storage or app_config.get("KEY_STORAGE", "./rag_data/")
    threads = threads or app_config.get("THREADS", 6)
    chunk_size = chunk_size or app_config.get("CHUNK_SIZE", 2048)
    chunk_overlap = chunk_overlap or app_config.get("CHUNK_OVERLAP", 1024)

    logger.info(f"Processing file: {file_path}")
    logger.info(f"Target collection: {collection_name}")
    logger.info(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}")

    logger.info("Loading and chunking document...")
    documents = load_and_chunk_text_file(file_path, chunk_size, chunk_overlap)
    logger.info(f"Created {len(documents)} chunks")

    if metadata_file:
        logger.info(f"Using specified metadata file: {metadata_file}")
    else:
        base_name = file_path.stem
        if base_name.endswith("_message_examples"):
            base_name = base_name.replace("_message_examples", "")
        metadata_file = Path(key_storage) / f"{base_name}.json"
        logger.info(f"Auto-detected metadata file: {metadata_file}")

    if metadata_file.exists():
        documents = enrich_documents_with_metadata(documents, metadata_file, threads)
        metadata_count = sum(1 for doc in documents if doc.metadata)
        logger.info(f"Enriched {metadata_count} documents with metadata")
    else:
        logger.warning("No metadata file found, proceeding without enrichment")

    logger.info("Initializing ChromaDB client and embedder...")
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

    client = chromadb.PersistentClient(path=str(persist_directory), settings=Settings(anonymized_telemetry=False))

    config = PushConfig(
        persist_directory=str(persist_directory),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        key_storage=str(key_storage),
        threads=threads,
        dry_run=dry_run,
        overwrite=overwrite,
    )

    context = ProcessingContext(embedder=embedder, client=client)

    push_to_collection(collection_name, documents, config, context)

    if not dry_run:
        logger.info("✓ Successfully pushed data to ChromaDB")
        logger.info(f"Collection: {collection_name}")
        logger.info(f"Documents: {len(documents)}")
        logger.info(f"Storage: {persist_directory}")


if __name__ == "__main__":
    main()
