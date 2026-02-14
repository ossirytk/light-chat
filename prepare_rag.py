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
class ProcessingConfig:
    """Configuration for processing text files to ChromaDB collections."""

    persist_directory: str
    chunk_size: int
    chunk_overlap: int
    key_storage: str
    threads: int


@dataclass
class ProcessingContext:
    embedder: HuggingFaceEmbeddings
    client: chromadb.PersistentClient


@dataclass
class CollectionJob:
    file_path: Path
    collection_name: str
    base_collection_name: str


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
    """Enrich a document with metadata keys found in its content.

    Expects `all_keys` to be a list of dicts where each dict contains a
    "uuid" and a text field (for example "text"). For each item, if the
    text appears in the document content, the document.metadata will be
    enriched with metadata[uuid] = text.
    """
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


def process_file_to_collection(
    job: CollectionJob,
    config: ProcessingConfig,
    context: ProcessingContext,
) -> None:
    """Process a single text file and save it to a ChromaDB collection."""
    logger.info(f"Processing {job.file_path} -> collection '{job.collection_name}'")

    # Load and chunk document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    loader = TextLoader(job.file_path, encoding="utf-8")
    documents = loader.load()
    all_documents = text_splitter.split_documents(documents)

    logger.info(f"Loaded {len(all_documents)} chunks from {job.file_path.name}")

    # Load metadata keys using base collection name
    key_storage_path = Path(config.key_storage) / f"{job.base_collection_name}.json"

    if key_storage_path.exists():
        logger.info(f"Loading filter list from: {key_storage_path}")
        with key_storage_path.open(encoding="utf-8") as key_file:
            all_keys = json.load(key_file)
        if "Content" in all_keys:
            all_keys = all_keys["Content"]

        # Enrich documents with metadata using multiprocessing
        logger.info(f"Enriching documents with metadata using {config.threads} threads")
        tic = time.perf_counter()

        enrich_fn = partial(enrich_document_with_metadata, all_keys=all_keys)
        with ProcessPoolExecutor(max_workers=config.threads) as executor:
            chunksize = max(1, len(all_documents) // config.threads)
            document_list = list(executor.map(enrich_fn, all_documents, chunksize=chunksize))

        toc = time.perf_counter()
        logger.info(f"Metadata enrichment took {toc - tic:0.4f} seconds")
    else:
        logger.warning(f"No keyfile found at {key_storage_path}, skipping metadata enrichment")
        document_list = all_documents

    # Store embeddings
    tic = time.perf_counter()
    Chroma.from_documents(
        client=context.client,
        documents=document_list,
        embedding=context.embedder,
        persist_directory=config.persist_directory,
        collection_name=job.collection_name,
        collection_metadata={"hnsw:space": "l2"},
    )
    toc = time.perf_counter()
    logger.info(f"Storing embeddings for '{job.collection_name}' took {toc - tic:0.4f} seconds")


@click.command()
@click.option(
    "--documents-directory",
    "-d",
    "documents_directory",
    default=None,
    help="The directory where your text files are stored",
)
@click.option(
    "--persist-directory",
    "-p",
    default=None,
    help="The directory where you want to store the Chroma collection.",
)
@click.option(
    "--key-storage",
    "-ks",
    default=None,
    help="The directory for the collection metadata keys.",
)
@click.option("--threads", "-t", default=None, type=int, help="The number of threads to use for parsing.")
@click.option("--chunk-size", "-cs", default=None, type=int, help="Data chunk for size for parsing.")
@click.option("--chunk-overlap", "-co", default=None, type=int, help="Overlap for the chunks.")
# TODO add option to specify the embedding model
def main(**kwargs: object) -> None:
    """
    This script parses text documents into chroma collections. Using langchain RecursiveSplitter.
    For each <collection_name>.txt file found, creates collection <collection_name>.
    For each <collection_name>_message_examples.txt file, creates collection <collection_name>_mes.
    Text documents are loaded from a directory and parsed into chunk sized text pieces.
    These pieces are matched for metadata keys in keyfile.
    The matching is done with multiprocess to improve perf for large collections and keyfiles.
    The resulting documents are pushed into Chroma vector data collections in persist-directory.
    """
    app_config = load_app_config()
    configure_logging(app_config)
    documents_directory = kwargs.get("documents_directory") or app_config.get("DOCUMENTS_DIRECTORY", "./rag_data/")
    persist_directory = kwargs.get("persist_directory") or app_config.get("PERSIST_DIRECTORY", "./character_storage/")
    key_storage = kwargs.get("key_storage") or app_config.get("KEY_STORAGE", "./rag_data/")
    threads = kwargs.get("threads") or app_config.get("THREADS", 6)
    chunk_size = kwargs.get("chunk_size") or app_config.get("CHUNK_SIZE", 2048)
    chunk_overlap = kwargs.get("chunk_overlap") or app_config.get("CHUNK_OVERLAP", 1024)
    docs_path = Path(str(documents_directory))
    all_txt_files = list(docs_path.glob("*.txt"))

    # Find base collection names (files that don't end with _message_examples.txt)
    base_files = [f for f in all_txt_files if not f.stem.endswith("_message_examples")]

    if not base_files:
        logger.warning(f"No base text files found in {documents_directory}")
        return

    logger.info(f"Found {len(base_files)} base collection(s) to process")

    # Initialize embedder and client once
    logger.info("Using huggingface embeddings")
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
    context = ProcessingContext(embedder=embedder, client=client)

    # Create processing configuration
    config = ProcessingConfig(
        persist_directory=str(persist_directory),
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        key_storage=str(key_storage),
        threads=int(threads),
    )

    # Process each base file and its corresponding message_examples file
    for base_file in base_files:
        collection_base_name = base_file.stem

        # Process base file -> <collection_name>
        process_file_to_collection(
            job=CollectionJob(
                file_path=base_file,
                collection_name=collection_base_name,
                base_collection_name=collection_base_name,
            ),
            config=config,
            context=context,
        )

        # Check for corresponding message_examples file
        message_examples_file = docs_path / f"{collection_base_name}_message_examples.txt"
        if message_examples_file.exists():
            # Process message_examples file -> <collection_name>_mes
            process_file_to_collection(
                job=CollectionJob(
                    file_path=message_examples_file,
                    collection_name=f"{collection_base_name}_mes",
                    base_collection_name=collection_base_name,
                ),
                config=config,
                context=context,
            )
        else:
            logger.info(f"No message_examples file found for {collection_base_name}")

    logger.info("All collections processed successfully")
    logger.info(f"Read files from directory: {documents_directory}")
    logger.info(f"Text parsed with chunk size: {chunk_size}, and chunk overlap: {chunk_overlap}")
    logger.info(f"Saved collections to: {persist_directory}")


if __name__ == "__main__":
    main()
