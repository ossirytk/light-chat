import json
import time
from concurrent.futures import ProcessPoolExecutor
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


def enrich_document_with_metadata(document: Document, all_keys:list) -> Document:
    """Enrich a document with metadata keys found in its content.

    Expects `all_keys` to be a list of dicts where each dict contains a
    "uuid" and a text field (for example "text"). For each item, if the
    text appears in the document content, the document.metadata will be
    enriched with metadata[uuid] = text.
    """
    if not isinstance(document.metadata, dict):
        document.metadata = {} if document.metadata is None else dict(document.metadata)

    if not isinstance(all_keys, list):
        return document

    text_keys = ("text", "text_fields", "text_field", "content", "value")
    for item in all_keys:
        if not isinstance(item, dict):
            continue
        uuid = item.get("uuid")
        if not isinstance(uuid, str):
            continue

        value = None
        for tk in text_keys:
            if tk in item and isinstance(item[tk], str):
                value = item[tk]
                break

        if value is None:
            for k, v in item.items():
                if k == "uuid":
                    continue
                if isinstance(v, str):
                    value = v
                    break

        if not isinstance(value, str):
            continue

        if value in document.page_content:
            document.metadata[uuid] = value

    return document


@click.command()
@click.option(
    "--documents-directory",
    "-d",
    "documents_directory",
    default="./rag_data/",
    help="The directory where your text files are stored",
)
@click.option("--collection-name", "-c", default="shodan", help="The name of the Chroma collection.")
@click.option(
    "--persist-directory",
    "-p",
    default="./character_storage/",
    help="The directory where you want to store the Chroma collection.",
)
@click.option(
    "--key-storage", "-ks", default="./rag_data/", help="The directory for the collection metadata keys.",
)
@click.option("--keyfile-name", "-kn", default="none", help="Keyfile name. If not given, defaults to collection name.")
@click.option("--threads", "-t", default=6, type=int, help="The number of threads to use for parsing.")
@click.option("--chunk-size", "-cs", default=2048, type=int, help="Data chunk for size for parsing.")
@click.option("--chunk-overlap", "-co", default=1024, type=int, help="Overlap for the chunks.")
# TODO add option to specify the embedding model
def main(
    documents_directory: str,
    collection_name: str,
    persist_directory: str,
    chunk_size: int,
    chunk_overlap: int,
    key_storage: str,
    keyfile_name: str,
    threads: int,
) -> None:
    """
    This script parses text documents into a chroma collection. Using langchain RecursiveSplitter.
    Text documents are loaded from a directory and parsed into chunk sized text pieces.
    These pieces are matched for metadata keys in keyfile.
    The matching is done with multiprocess to improve perf for large collections and keyfiles.
    The resulting documents are pushed into a Chroma vector data collection in persist-directory.
    """
    documents_paths_txt = list(Path(documents_directory).glob("*.txt"))

    # Load and chunk documents
    logger.info(f"Loading text files from: {documents_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_documents = []
    for txt_document in documents_paths_txt:
        loader = TextLoader(txt_document, encoding="utf-8")
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        all_documents.extend(docs)

    logger.info(f"Loaded {len(all_documents)} chunks")

    # Load metadata keys
    if keyfile_name == "none":
        key_storage_path = str(Path(key_storage) / (collection_name + ".json"))
    else:
        key_storage_path = str(Path(key_storage) / keyfile_name)

    logger.info(f"Loading filter list from: {key_storage_path}")
    with Path(key_storage_path).open(encoding="utf-8") as key_file:
        all_keys = json.load(key_file)
    if "Content" in all_keys:
        all_keys = all_keys["Content"]

    # Enrich documents with metadata using multiprocessing
    logger.info(f"Enriching documents with metadata using {threads} threads")
    tic = time.perf_counter()

    enrich_fn = partial(enrich_document_with_metadata, all_keys=all_keys)
    with ProcessPoolExecutor(max_workers=threads) as executor:
        document_list = list(executor.map(enrich_fn, all_documents, chunksize=max(1, len(all_documents) // threads)))

    # Stop timer
    toc = time.perf_counter()
    logger.info(f"Metadata enrichment took {toc - tic:0.4f} seconds")

    tic = time.perf_counter()


    logger.info("Using huggingface embeddings")
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    cache_folder = str(Path("./embedding_models"))
    embedder = HuggingFaceEmbeddings(
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder=cache_folder,
    )

    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
    Chroma.from_documents(
        client=client,
        documents=document_list,
        embedding=embedder,
        persist_directory=persist_directory,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "l2"},
    )

    # Stop timer
    toc = time.perf_counter()
    logger.info(f"Storing embeddings took {toc - tic:0.4f} seconds")

    logger.info(f"Read metadata filters from directory: {key_storage_path}")
    logger.info(f"Read files from directory: {documents_directory}")
    logger.info(f"Text parsed with chunk size: {chunk_size}, and chunk overlap: {chunk_overlap}")
    logger.info(f"Saved collection as: {collection_name}")
    logger.info(f"Saved collection to: {persist_directory}")


if __name__ == "__main__":
    main()
