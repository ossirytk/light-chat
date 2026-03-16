"""Migration CLI commands for re-embedding collections with a new model."""

from pathlib import Path

import click

from core.config import load_app_config, load_rag_script_config
from scripts.rag.migrate_collection_embedding import (
    MigrationConfig,
    MigrationSpec,
    format_migration_report,
    run_migration,
    write_migration_report_csv,
    write_migration_report_json,
)


@click.group("migrate")
def migrate_group() -> None:
    """Re-embed collections with a new embedding model."""


@migrate_group.command("embedding")
@click.option(
    "--collection",
    "collections",
    multiple=True,
    required=True,
    metavar="NAME",
    help="Collection name to migrate; repeatable.",
)
@click.option(
    "--target-model",
    required=True,
    help="HuggingFace model ID to re-embed with.",
)
@click.option("--device", default="cpu", show_default=True, help="Device for the target model.")
@click.option(
    "--no-normalize",
    "normalize",
    is_flag=True,
    default=True,
    help="Disable embedding normalization (normalised by default).",
)
@click.option(
    "--persist-directory",
    "-p",
    default=None,
    help="Directory where ChromaDB stores collections. Falls back to app config.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Build temp collections but skip the destructive rename/delete.",
)
@click.option(
    "--fixture-file",
    "fixture_file",
    type=click.Path(path_type=Path),
    default=None,
    help="Fixture JSON for post-migration validation. Migration aborts if Recall@k is below threshold.",
)
@click.option(
    "--validation-threshold",
    type=float,
    default=0.75,
    show_default=True,
    help="Minimum Recall@k required to commit migration (requires --fixture-file).",
)
@click.option(
    "--output-json",
    type=click.Path(path_type=Path),
    default=None,
    help="Write migration report as JSON.",
)
@click.option(
    "--output-csv",
    type=click.Path(path_type=Path),
    default=None,
    help="Write migration report as CSV.",
)
def migrate_embedding(**kwargs: object) -> None:
    """Re-embed one or more collections with a new embedding model.

    Uses an atomic alias-swap: builds a temp collection with the new model,
    optionally validates it against a retrieval fixture, then deletes the old
    collection and renames the temp to take its place.

    Example::

        python -m scripts.rag.manage_collections migrate embedding \\
            --collection shodan --collection shodan_mes \\
            --target-model sentence-transformers/all-MiniLM-L6-v2 \\
            --dry-run
    """
    collections: tuple[str, ...] = kwargs["collections"]  # type: ignore[assignment]
    target_model = str(kwargs["target_model"])
    device = str(kwargs["device"])
    normalize = bool(kwargs["normalize"])
    persist_directory: str | None = kwargs["persist_directory"]  # type: ignore[assignment]
    dry_run = bool(kwargs["dry_run"])
    fixture_file: Path | None = kwargs["fixture_file"]  # type: ignore[assignment]
    validation_threshold = float(kwargs["validation_threshold"])
    output_json: Path | None = kwargs["output_json"]  # type: ignore[assignment]
    output_csv: Path | None = kwargs["output_csv"]  # type: ignore[assignment]

    specs = [
        MigrationSpec(
            collection_name=name,
            target_model_id=target_model,
            target_device=device,
            target_normalize=normalize,
        )
        for name in collections
        if name.strip()
    ]

    if not specs:
        msg = "Provide at least one --collection name"
        raise click.UsageError(msg)

    app_config = load_app_config()
    script_config = load_rag_script_config(app_config)
    config = MigrationConfig(
        persist_directory=persist_directory or script_config.persist_directory,
        embedding_cache=script_config.embedding_cache,
        dry_run=dry_run,
        fixture_file=fixture_file,
        validation_threshold=validation_threshold,
    )

    results = run_migration(specs, config)
    click.echo("\n" + format_migration_report(results))

    if output_json is not None:
        write_migration_report_json(output_json, results)
        click.echo(f"Wrote JSON report: {output_json}")
    if output_csv is not None:
        write_migration_report_csv(output_csv, results)
        click.echo(f"Wrote CSV report: {output_csv}")

    failed = sum(1 for r in results if not r.succeeded)
    if failed:
        msg = f"{failed} collection(s) failed migration"
        raise click.ClickException(msg)


def register_migration_commands(cli: click.Group) -> None:
    """Attach migration commands to a CLI group."""
    cli.add_command(migrate_group)
