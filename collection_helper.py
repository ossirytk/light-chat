from pathlib import Path

import chromadb
import click
from chromadb.api.client import Client
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger


@click.command()
@click.argument("command", type=click.Choice(["list", "delete", "test"]))
@click.option(
    "--collection-name",
    "-c",
    "collection_name",
    default="shodan",
    help="The name of the Chroma collection that's the target of an action",
)
@click.option(
    "--persist-directory",
    "-p",
    "persist_directory",
    default="./character_storage/",
    help="The directory where you want to store the Chroma collection",
)
# TODO add option to specify the embedding model
def main(
    collection_name: str,
    persist_directory: str,
    command: str,
) -> None:
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    cache_folder = str(Path("./embedding_models"))
    embedder = HuggingFaceEmbeddings(
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder=cache_folder,
    )
    client: Client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
    db = Chroma(
        client=client,
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedder,
    )
    match command:
        case "list":
            logger.info("Available collections:")
            collections = client.list_collections()
            for collection in collections:
                logger.info(collection.name)
        case "delete":
            logger.info(f"Deleting {collection_name}")
            client.delete_collection(collection_name)
            logger.info(f"{collection_name} deleted")
        case "test":
            logger.info("Testing fetch")
            k_buffer = 7
            """
            where={
                    "$and": [
                        {'1ae139a7-8d9f-4c75-8167-84f1a6583593': 'Edward Diego'},
                        {'46a6a095-52c5-4a64-9c3d-bd7362261e09': 'Cyberspace'},
                    ],
                }
            """
            where={'1ae139a7-8d9f-4c75-8167-84f1a6583593': 'Edward Diego'}
            docs = db.similarity_search_with_score(query="Shodan talks to Diego", k=k_buffer, filter=where)
            logger.info(f"Fetched {len(docs)} documents")
            print(docs[0])
        case _:
            collections = client.list_collections()
            logger.info("Available collections:")
            for collection in collections:
                logger.info(collection.name)


if __name__ == "__main__":
    main()
