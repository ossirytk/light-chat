"""Collection management CLI commands."""

import fnmatch
import json
from pathlib import Path

import chromadb
import click
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import load_app_config, load_rag_script_config
from scripts.rag.manage_collections_core import (
    _resolve_embedding_runtime,
    assert_collection_fingerprint_compatible,
    build_embedding_fingerprint,
    build_where_filters,
    extract_key_matches,
    get_collection_info,
    infer_embedding_dimension,
    normalize_keyfile,
)


def _load_key_filters(key_storage: str, collection_name: str, query: str) -> list[dict[str, object] | None]:
    base_name = collection_name.replace("_mes", "")
    keyfile_path = Path(key_storage) / f"{base_name}.json"
    filters: list[dict[str, object] | None] = [None]

    if not keyfile_path.exists():
        return filters

    click.echo(f"Loading metadata from: {keyfile_path}")
    with keyfile_path.open(encoding="utf-8") as f:
        key_data = json.load(f)
    keys = normalize_keyfile(key_data)
    matches = extract_key_matches(keys, query)
    if matches:
        click.echo(f"Matched {len(matches)} metadata key(s) from query")
        filters = build_where_filters(matches)
    return filters


def _print_test_result(docs: list[tuple[object, float]]) -> None:
    click.echo(f"\N{CHECK MARK} Found {len(docs)} result(s)")
    for idx, (doc, score) in enumerate(docs, 1):
        page_content = str(getattr(doc, "page_content", ""))
        metadata = getattr(doc, "metadata", None)
        click.echo(f"\nResult {idx} (score: {score:.4f}):")
        click.echo(f"Content preview: {page_content[:200]}...")
        if metadata:
            click.echo(f"Metadata keys: {list(metadata.keys())[:5]}")


def _run_test_searches(
    db: Chroma,
    query: str,
    k: int,
    filters: list[dict[str, object] | None],
) -> bool:
    for filter_idx, where_filter in enumerate(filters):
        filter_label = "unfiltered" if where_filter is None else str(where_filter)
        click.echo(f"\nAttempting search #{filter_idx + 1} ({filter_label})")
        docs = (
            db.similarity_search_with_score(query=query, k=k)
            if where_filter is None
            else db.similarity_search_with_score(query=query, k=k, filter=where_filter)
        )

        if docs:
            _print_test_result(docs)
            return True
    return False


@click.command()
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
    script_config = load_rag_script_config(app_config)
    persist_directory = persist_directory or script_config.persist_directory

    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))

    collections = client.list_collections()

    if not collections:
        click.echo("No collections found")
        return

    click.echo(f"Found {len(collections)} collection(s):")

    for collection in collections:
        if verbose:
            info = get_collection_info(client, collection.name)
            click.echo(f"  \N{BULLET} {info['name']}")
            click.echo(f"    - Documents: {info['count']}")
            click.echo(f"    - Metadata: {info['metadata']}")
        else:
            click.echo(f"  \N{BULLET} {collection.name} ({collection.count()} documents)")


@click.command()
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
    script_config = load_rag_script_config(app_config)
    persist_directory = persist_directory or script_config.persist_directory

    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))

    info = get_collection_info(client, collection_name)

    if not info["exists"]:
        click.secho(f"Collection '{collection_name}' not found", fg="red", err=True)
        return

    if not yes:
        click.secho(f"About to delete collection: {collection_name}", fg="yellow")
        click.secho(f"This collection contains {info['count']} documents", fg="yellow")
        confirmation = input("Are you sure? (yes/no): ")
        if confirmation.lower() not in ["yes", "y"]:
            click.echo("Deletion cancelled")
            return

    try:
        client.delete_collection(collection_name)
        click.echo(f"\N{CHECK MARK} Deleted collection: {collection_name}")
    except ValueError as e:
        click.secho(f"Error deleting collection: {e}", fg="red", err=True)


@click.command()
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
    script_config = load_rag_script_config(app_config)
    persist_directory = persist_directory or script_config.persist_directory

    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))

    collections = client.list_collections()

    if not pattern:
        click.secho("Must specify --pattern for bulk deletion", fg="red", err=True)
        return

    matching = [c for c in collections if fnmatch.fnmatch(c.name, pattern)]

    if not matching:
        click.echo(f"No collections match pattern: {pattern}")
        return

    click.echo(f"Found {len(matching)} collection(s) matching pattern '{pattern}':")
    for collection in matching:
        click.echo(f"  \N{BULLET} {collection.name} ({collection.count()} documents)")

    if not yes:
        confirmation = input(f"Delete all {len(matching)} collections? (yes/no): ")
        if confirmation.lower() not in ["yes", "y"]:
            click.echo("Deletion cancelled")
            return

    for collection in matching:
        try:
            client.delete_collection(collection.name)
            click.echo(f"\N{CHECK MARK} Deleted: {collection.name}")
        except ValueError as e:
            click.secho(f"\N{BALLOT X} Error deleting {collection.name}: {e}", fg="red", err=True)


@click.command()
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
@click.option("--embedding-model", default=None, help="Override embedding model id for this test run")
@click.option("--embedding-device", default=None, help="Override embedding device for this test run")
def test(**kwargs: object) -> None:
    """Test a collection with a similarity search query."""
    app_config = load_app_config()
    script_config = load_rag_script_config(app_config)
    collection_name = str(kwargs["collection_name"])
    query = str(kwargs["query"])
    k = int(kwargs["k"])
    persist_directory = str(kwargs.get("persist_directory") or script_config.persist_directory)
    key_storage = str(kwargs.get("key_storage") or script_config.key_storage)
    embedding_model = str(kwargs.get("embedding_model") or script_config.embedding_model)
    embedding_device = str(kwargs.get("embedding_device") or script_config.embedding_device)
    embedding_cache = script_config.embedding_cache
    normalize_embeddings = True

    embedder = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": embedding_device},
        encode_kwargs={"normalize_embeddings": normalize_embeddings},
        cache_folder=str(Path(embedding_cache)),
    )

    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
    expected_fingerprint = build_embedding_fingerprint(
        embedding_model=embedding_model,
        normalize_embeddings=normalize_embeddings,
        embedding_dimension=infer_embedding_dimension(embedder),
    )
    assert_collection_fingerprint_compatible(client, collection_name, expected_fingerprint)

    info = get_collection_info(client, collection_name)
    if not info["exists"]:
        click.secho(f"Collection '{collection_name}' not found", fg="red", err=True)
        return

    click.echo(f"Testing collection: {collection_name}")
    click.echo(f"Query: {query}")
    click.echo(f"Total documents: {info['count']}")

    filters = _load_key_filters(key_storage, collection_name, query)

    db = Chroma(
        client=client,
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedder,
    )

    if _run_test_searches(db, query, k, filters):
        return

    click.echo("No results found")


@click.command()
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
    script_config = load_rag_script_config(app_config)
    persist_directory = persist_directory or script_config.persist_directory

    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))

    info = get_collection_info(client, collection_name)
    if not info["exists"]:
        click.secho(f"Collection '{collection_name}' not found", fg="red", err=True)
        return

    click.echo(f"Exporting collection: {collection_name}")

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

    click.echo(f"\N{CHECK MARK} Exported {export_data['count']} documents to {output}")


@click.command()
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
    script_config = load_rag_script_config(app_config)
    persist_directory = persist_directory or script_config.persist_directory

    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))

    info = get_collection_info(client, collection_name)

    if not info["exists"]:
        click.secho(f"Collection '{collection_name}' not found", fg="red", err=True)
        return

    collection = client.get_collection(collection_name)
    sample = collection.peek(limit=1)

    click.echo(f"Collection: {collection_name}")
    click.echo(f"Documents: {info['count']}")
    click.echo(f"Metadata: {info['metadata']}")

    if sample.get("documents"):
        click.echo("\nSample document:")
        click.echo(f"  Content: {sample['documents'][0][:150]}...")
        if sample.get("metadatas") and sample["metadatas"][0]:
            click.echo(f"  Metadata keys: {list(sample['metadatas'][0].keys())}")


@click.command("backfill-embedding-fingerprint")
@click.option(
    "--persist-directory",
    "-p",
    default=None,
    help="Directory where ChromaDB stores collections",
)
@click.option("--pattern", default="*", show_default=True, help="Collection name filter (wildcards allowed)")
@click.option("--embedding-model", default=None, help="Embedding model id to stamp as fingerprint")
@click.option("--embedding-device", default=None, help="Embedding device used to infer embedding dimension")
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite conflicting fingerprint keys when collections already have different values",
)
@click.option("--dry-run", is_flag=True, help="Preview metadata updates without writing changes")
def backfill_embedding_fingerprint(**kwargs: object) -> None:
    """Backfill embedding fingerprint metadata on existing collections.

    This command is intended for migrating legacy collections that were created
    before embedding fingerprint metadata was enforced.
    """
    app_config = load_app_config()
    script_config = load_rag_script_config(app_config)
    persist_directory = kwargs.get("persist_directory") or script_config.persist_directory
    pattern = str(kwargs.get("pattern") or "*")
    force = bool(kwargs.get("force"))
    dry_run = bool(kwargs.get("dry_run"))

    embedding_model, embedding_device, embedding_cache = _resolve_embedding_runtime(
        embedding_model=kwargs.get("embedding_model"),
        embedding_device=kwargs.get("embedding_device"),
    )
    embedder = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": embedding_device},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=str(Path(embedding_cache)),
    )
    expected_fingerprint = build_embedding_fingerprint(
        embedding_model=embedding_model,
        normalize_embeddings=True,
        embedding_dimension=infer_embedding_dimension(embedder),
    )

    client = chromadb.PersistentClient(path=str(persist_directory), settings=Settings(anonymized_telemetry=False))
    all_collections = client.list_collections()
    matching = [collection for collection in all_collections if fnmatch.fnmatch(collection.name, pattern)]

    if not matching:
        click.echo(f"No collections match pattern: {pattern}")
        return

    updated = 0
    unchanged = 0
    skipped_conflict = 0

    click.echo(f"Backfilling embedding fingerprint for {len(matching)} collection(s)")
    click.echo(f"Fingerprint: {expected_fingerprint}")
    if dry_run:
        click.echo("Mode: dry-run (no writes)")

    for collection_info in matching:
        collection = client.get_collection(collection_info.name)
        metadata = {
            key: value
            for key, value in dict(collection.metadata or {}).items()
            if key != "hnsw:space"
        }

        mismatches = [
            (key, metadata[key], expected_fingerprint[key])
            for key in expected_fingerprint
            if key in metadata and metadata[key] != expected_fingerprint[key]
        ]
        if mismatches and not force:
            mismatch_summary = ", ".join(
                f"{key}: existing={actual!r} expected={expected!r}" for key, actual, expected in mismatches
            )
            click.secho(f"- {collection.name}: skipped (conflict: {mismatch_summary})", fg="yellow")
            skipped_conflict += 1
            continue

        new_metadata = {**metadata, **expected_fingerprint}
        if new_metadata == metadata:
            click.echo(f"- {collection.name}: unchanged")
            unchanged += 1
            continue

        if dry_run:
            click.secho(f"- {collection.name}: would update metadata", fg="green")
            updated += 1
            continue

        collection.modify(metadata=new_metadata)
        click.secho(f"- {collection.name}: updated metadata", fg="green")
        updated += 1

    click.echo("\nSummary:")
    click.echo(f"  matched: {len(matching)}")
    click.echo(f"  updated: {updated}")
    click.echo(f"  unchanged: {unchanged}")
    click.echo(f"  skipped_conflict: {skipped_conflict}")


def register_collection_commands(cli: click.Group) -> None:
    """Attach collection-management commands to a CLI group."""
    cli.add_command(list_collections)
    cli.add_command(delete)
    cli.add_command(delete_multiple)
    cli.add_command(test)
    cli.add_command(export)
    cli.add_command(info)
    cli.add_command(backfill_embedding_fingerprint)
