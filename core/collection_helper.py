import json
from dataclasses import dataclass
from pathlib import Path

import chromadb
import click
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_app_config() -> dict:
    config_path = Path("./configs/") / "appconf.json"
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        return json.load(f)


@dataclass
class TestContext:
    client: chromadb.PersistentClient
    persist_directory: str
    embedder: HuggingFaceEmbeddings
    key_storage: str


def create_db(
    client: chromadb.PersistentClient,
    persist_directory: str,
    embedder: HuggingFaceEmbeddings,
    target_collection: str,
) -> Chroma:
    return Chroma(
        client=client,
        collection_name=target_collection,
        persist_directory=persist_directory,
        embedding_function=embedder,
    )


def normalize_keyfile(raw_keys: object) -> list[dict[str, object]]:
    if isinstance(raw_keys, dict) and "Content" in raw_keys:
        raw_keys = raw_keys["Content"]
    if not isinstance(raw_keys, list):
        return []
    return [item for item in raw_keys if isinstance(item, dict)]


def _get_entry_value(item: dict[str, object]) -> str | None:
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


def _matches_aliases(item: dict[str, object], text_lower: str) -> bool:
    aliases = item.get("aliases")
    if not isinstance(aliases, list):
        return False
    return any(isinstance(a, str) and a.lower() in text_lower for a in aliases)


def extract_key_matches(keys: list[dict[str, object]], text: str) -> list[dict[str, str]]:
    if not text:
        return []
    text_lower = text.lower()
    matches: list[dict[str, str]] = []
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


def build_where_filters(matches: list[dict[str, str]]) -> list[dict[str, object] | None]:
    if not matches:
        return [None]
    if len(matches) == 1:
        return [matches[0]]
    return [{"$and": matches}, {"$or": matches}]


def run_test_search(
    target_collection: str,
    query: str,
    filters: list[dict[str, object] | None],
    context: TestContext,
) -> None:
    db = create_db(context.client, context.persist_directory, context.embedder, target_collection)
    k_buffer = 7
    for filter_idx, where in enumerate(filters):
        filter_label = "unfiltered" if where is None else str(where)
        click.echo(f"{target_collection}: attempting search #{filter_idx + 1} ({filter_label})")
        if where is None:
            docs = db.similarity_search_with_score(query=query, k=k_buffer)
        else:
            docs = db.similarity_search_with_score(query=query, k=k_buffer, filter=where)
        if docs or where is None:
            click.echo(f"{target_collection}: fetched {len(docs)} documents")
            if docs:
                click.echo(str(docs[0]))
            return
    click.echo(f"{target_collection}: fetched 0 documents")


def run_test_command(
    collection_name: str,
    query: str,
    context: TestContext,
) -> None:
    click.echo("Testing fetch")
    if not query:
        click.secho("No query provided. Use --query to supply a search string.", fg="yellow")
        return

    keyfile_path = Path(context.key_storage) / f"{collection_name}.json"
    if not keyfile_path.exists():
        click.secho(f"Keyfile not found at {keyfile_path}. Using unfiltered search.", fg="yellow")
        filters = [None]
    else:
        with keyfile_path.open(encoding="utf-8") as key_file:
            key_data = json.load(key_file)
        keys = normalize_keyfile(key_data)
        matches = extract_key_matches(keys, query)
        if matches:
            click.echo(f"Matched metadata keys from query: {matches}")
        else:
            click.echo("No metadata keys matched the query")
        filters = build_where_filters(matches)

    run_test_search(collection_name, query, filters, context)
    run_test_search(f"{collection_name}_mes", query, filters, context)


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
    default=None,
    help="The directory where you want to store the Chroma collection",
)
@click.option(
    "--key-storage",
    "-ks",
    "key_storage",
    default=None,
    help="The directory where the keyfile JSON is stored",
)
@click.option(
    "--query",
    "-q",
    "query",
    default="",
    help="Query text to search the collection(s) during test",
)
# TODO add option to specify the embedding model
def main(
    collection_name: str,
    persist_directory: str,
    key_storage: str,
    query: str,
    command: str,
) -> None:
    app_config = load_app_config()
    if persist_directory is None:
        persist_directory = app_config.get("PERSIST_DIRECTORY", "./character_storage/")
    if key_storage is None:
        key_storage = app_config.get("KEY_STORAGE", "./rag_data/")
    embedding_device = str(app_config.get("EMBEDDING_DEVICE", "cpu"))
    embedding_cache = str(app_config.get("EMBEDDING_CACHE", "./embedding_models/"))
    model_kwargs = {"device": embedding_device}
    encode_kwargs = {"normalize_embeddings": True}
    cache_folder = str(Path(embedding_cache))
    embedder = HuggingFaceEmbeddings(
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder=cache_folder,
    )
    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
    context = TestContext(
        client=client,
        persist_directory=persist_directory,
        embedder=embedder,
        key_storage=key_storage,
    )
    match command:
        case "list":
            collections = client.list_collections()
            if not collections:
                click.echo("No collections found")
            else:
                click.echo("Available collections:")
                for collection in collections:
                    click.echo(collection.name)
        case "delete":
            click.echo(f"Deleting {collection_name}")
            client.delete_collection(collection_name)
            click.echo(f"{collection_name} deleted")
        case "test":
            run_test_command(collection_name, query, context)
        case _:
            collections = client.list_collections()
            if not collections:
                click.echo("No collections found")
            else:
                click.echo("Available collections:")
                for collection in collections:
                    click.echo(collection.name)


if __name__ == "__main__":
    main()
