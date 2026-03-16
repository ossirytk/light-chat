"""Enhanced script for pushing RAG data to ChromaDB.

This script provides comprehensive features for managing RAG data:
- Batch upload of multiple text files
- Metadata enrichment with validation
- Collection versioning support
- Progress tracking and error handling
- Dry-run mode for testing
"""

import json
import multiprocessing as mp
import re
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

from core.config import configure_logging, load_app_config, load_rag_script_config
from scripts.rag.analyze_rag_coverage import extract_coverage_metrics, format_coverage_report, load_metadata_file

type MetadataItem = dict[str, object]
type MetadataList = list[MetadataItem]

EMBEDDING_MODEL_METADATA_KEY = "embedding:model"
EMBEDDING_DIMENSION_METADATA_KEY = "embedding:dimension"
EMBEDDING_NORMALIZE_METADATA_KEY = "embedding:normalize"


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


def infer_embedding_dimension(embedder: HuggingFaceEmbeddings) -> int | None:
    try:
        vector = embedder.embed_query("dimension_probe")
    except Exception as error:
        logger.warning(f"Could not infer embedding dimension for fingerprint metadata: {error}")
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
    except ValueError:
        return

    metadata = collection.metadata or {}
    missing_keys = [key for key in expected_fingerprint if key not in metadata]
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
            f"Collection '{collection_name}' has incompatible embedding fingerprint; refusing mixed-model write. "
            f"{mismatch_summary}"
        )
        raise click.ClickException(msg)

    if missing_keys:
        logger.warning(
            "Collection '{}' is missing embedding fingerprint metadata keys: {}",
            collection_name,
            ", ".join(missing_keys),
        )


def iter_key_items(all_keys: object) -> MetadataList:
    if not isinstance(all_keys, list):
        return []
    return [item for item in all_keys if isinstance(item, dict)]


def get_item_value(item: MetadataItem, text_keys: tuple[str, ...]) -> str | None:
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


def enrich_document_with_metadata(document: Document, all_keys: MetadataList) -> Document:
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


_LEADING_HTML_COMMENT_RE = re.compile(r"^\s*<!--.*?-->\s*", re.DOTALL)


def strip_leading_html_comment(text: str) -> str:
    """Strip a leading HTML comment block from document text.

    Removes the first ``<!-- ... -->`` block at the start of the text so that
    document header metadata is not embedded as retrievable content.
    """
    return _LEADING_HTML_COMMENT_RE.sub("", text, count=1)


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
    for doc in documents:
        doc.page_content = strip_leading_html_comment(doc.page_content)
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
    with ProcessPoolExecutor(max_workers=threads, mp_context=mp.get_context("spawn")) as executor:
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
    expected_fingerprint: dict[str, object],
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
    else:
        assert_collection_fingerprint_compatible(context.client, collection_name, expected_fingerprint)

    logger.info(f"Pushing {len(documents)} documents to collection '{collection_name}'")
    tic = time.perf_counter()

    Chroma.from_documents(
        client=context.client,
        documents=documents,
        embedding=context.embedder,
        persist_directory=config.persist_directory,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "l2", **expected_fingerprint},
    )

    toc = time.perf_counter()
    logger.info(f"Collection '{collection_name}' created in {toc - tic:0.4f} seconds")


@dataclass
class CliOptions:
    """CLI options for push_rag_data."""

    file_path: Path
    collection_name: str
    persist_directory: str | None
    key_storage: str | None
    metadata_file: Path | None
    chunk_size: int | None
    chunk_overlap: int | None
    threads: int | None
    dry_run: bool
    overwrite: bool
    coverage_threshold: float = 0.75
    force_low_coverage: bool = False


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
@click.option("--embedding-model", default=None, help="Override embedding model id for this run")
@click.option("--embedding-device", default=None, help="Override embedding device for this run")
@click.option("--dry-run", "-d", is_flag=True, help="Show what would be done without making changes")
@click.option("--overwrite", "-w", is_flag=True, help="Overwrite existing collection if it exists")
@click.option(
    "--coverage-threshold",
    type=float,
    default=0.75,
    help="Minimum source coverage ratio (0.0-1.0) required for push; default 0.75",
)
@click.option(
    "--force-low-coverage",
    is_flag=True,
    help="Bypass coverage threshold check if below minimum",
)
@click.option(
    "--category-confidence-threshold",
    type=float,
    default=0.75,
    help="Minimum confidence required to assign entity category (0.0-1.0); default 0.75",
)
@click.option(
    "--allow-unassigned-categories",
    is_flag=True,
    help="If set, entities below confidence threshold get category=null instead of fallback",
)
def main(**kwargs: object) -> None:  # noqa: PLR0915
    """Push a text file to ChromaDB with metadata enrichment.

    This script provides enhanced features over prepare_rag.py:
    - Single file processing with explicit collection naming
    - Dry-run mode for testing
    - Overwrite protection
    - Custom metadata file selection
    - Detailed progress tracking
    """
    app_config = load_app_config()
    script_config = load_rag_script_config(app_config)
    configure_logging(app_config)

    file_path = kwargs.get("file_path")
    collection_name = kwargs.get("collection_name")
    persist_directory = kwargs.get("persist_directory") or script_config.persist_directory
    key_storage = kwargs.get("key_storage") or script_config.key_storage
    metadata_file = kwargs.get("metadata_file")
    threads = kwargs.get("threads") or script_config.threads
    chunk_size = kwargs.get("chunk_size") or script_config.chunk_size
    chunk_overlap = kwargs.get("chunk_overlap") or script_config.chunk_overlap
    dry_run = kwargs.get("dry_run", False)
    overwrite = kwargs.get("overwrite", False)
    coverage_threshold = kwargs.get("coverage_threshold", 0.75)
    force_low_coverage = kwargs.get("force_low_coverage", False)
    category_confidence_threshold = kwargs.get("category_confidence_threshold", 0.75)
    allow_unassigned_categories = kwargs.get("allow_unassigned_categories", False)

    # Category flags are applied at analyze-time; log them for visibility only
    logger.debug(
        "Category config (applies at analyze-time): threshold={}, allow_unassigned={}",
        category_confidence_threshold,
        allow_unassigned_categories,
    )

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

    # Phase 3b: Coverage quality gate
    if metadata_file.exists():
        try:
            source_text = file_path.read_text(encoding="utf-8")
            metadata = load_metadata_file(metadata_file)
            metrics = extract_coverage_metrics(source_text, metadata)
            report = format_coverage_report(metrics, coverage_threshold)
            logger.info(f"\nCoverage Quality Gate:\n{report}")

            if metrics.source_coverage_ratio < coverage_threshold:
                if not force_low_coverage:
                    msg = (
                        f"Source coverage {metrics.source_coverage_ratio * 100:.1f}% "
                        f"below threshold {coverage_threshold * 100:.0f}%. "
                        f"Pass --force-low-coverage to override."
                    )
                    raise click.ClickException(msg)  # noqa: TRY301
                logger.warning("Coverage below threshold but --force-low-coverage flag is set, proceeding")
        except click.ClickException:
            raise
        except Exception as e:
            logger.warning(f"Could not compute coverage metrics: {e}")

    logger.info("Initializing ChromaDB client and embedder...")
    embedding_device = kwargs.get("embedding_device") or script_config.embedding_device
    embedding_model = kwargs.get("embedding_model") or script_config.embedding_model
    embedding_cache = script_config.embedding_cache
    model_kwargs = {"device": embedding_device}
    normalize_embeddings = True
    encode_kwargs = {"normalize_embeddings": normalize_embeddings}
    cache_folder = str(Path(embedding_cache))

    embedder = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder=cache_folder,
    )

    client = chromadb.PersistentClient(path=str(persist_directory), settings=Settings(anonymized_telemetry=False))
    embedding_dimension = infer_embedding_dimension(embedder)
    expected_fingerprint = build_embedding_fingerprint(
        embedding_model=embedding_model,
        normalize_embeddings=normalize_embeddings,
        embedding_dimension=embedding_dimension,
    )

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

    push_to_collection(collection_name, documents, config, context, expected_fingerprint)

    if not dry_run:
        logger.info("✓ Successfully pushed data to ChromaDB")
        logger.info(f"Collection: {collection_name}")
        logger.info(f"Documents: {len(documents)}")
        logger.info(f"Storage: {persist_directory}")


if __name__ == "__main__":
    main()
