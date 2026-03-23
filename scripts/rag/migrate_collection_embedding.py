"""Re-embedding migration orchestration.

Migrates one or more ChromaDB collections from an old embedding model to a new one using an
atomic alias-swap pattern:

1. Fetch all document texts + metadata from the source collection.
2. Build a temporary collection (``{name}_mig_<ts>``) using the new embedder.
3. Optionally validate against a retrieval fixture (Recall@k gate).
4. On success: delete the source collection, rename temp → target name.
5. On failure or dry-run: clean up temp and leave source unchanged.

Dry-run mode completes steps 1-2 but skips the destructive rename/delete.
"""

import csv
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
import click
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from core.config import load_app_config, load_rag_script_config
from scripts.rag.manage_collections_core_collection import (
    build_embedding_fingerprint,
    infer_embedding_dimension,
)
from scripts.rag.manage_collections_core_evaluation import _execute_fixture_evaluation
from scripts.rag.manage_collections_core_types import (
    EMBEDDING_DIMENSION_METADATA_KEY,
    EMBEDDING_MODEL_METADATA_KEY,
    EMBEDDING_NORMALIZE_METADATA_KEY,
    FixtureEvalOptions,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_MIGRATION_SUFFIX = "mig"
_EMBED_BATCH_SIZE = 256
_DEFAULT_VALIDATION_THRESHOLD = 0.75
_FINGERPRINT_KEYS = frozenset({
    EMBEDDING_MODEL_METADATA_KEY,
    EMBEDDING_NORMALIZE_METADATA_KEY,
    EMBEDDING_DIMENSION_METADATA_KEY,
})


# ── Data Types ────────────────────────────────────────────────────────────────


@dataclass
class MigrationSpec:
    """Specification for migrating a single collection to a new embedding model."""

    collection_name: str
    target_model_id: str
    target_device: str = "cpu"
    target_normalize: bool = True


@dataclass
class MigrationResult:
    """Outcome record for a single collection migration."""

    collection_name: str
    source_model: str
    target_model: str
    docs_migrated: int
    status: str  # "success" | "dry_run" | "validation_failed" | "error"
    validation_recall: float | None = None
    error: str | None = None
    generated_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())

    @property
    def succeeded(self) -> bool:
        return self.status in ("success", "dry_run")


@dataclass
class _CollectionDocs:
    """Document payload fetched from a persistent collection."""

    texts: list[str]
    metadatas: list[dict[str, Any]]
    ids: list[str]


@dataclass
class _ValidationOptions:
    """Options for the post-build validation gate."""

    fixture_file: Path
    persist_directory: str
    threshold: float


@dataclass
class MigrationConfig:
    """Operational config for a migration run."""

    persist_directory: str
    embedding_cache: str
    dry_run: bool = False
    fixture_file: Path | None = None
    validation_threshold: float = _DEFAULT_VALIDATION_THRESHOLD


# ── Document Extraction ───────────────────────────────────────────────────────


def _fetch_collection_documents(
    client: chromadb.PersistentClient,
    collection_name: str,
) -> _CollectionDocs:
    """Return document texts, metadata, and ids from the named persistent collection.

    Raises ``click.ClickException`` if the collection does not exist.
    """
    try:
        raw_col = client.get_collection(collection_name)
    except Exception as exc:
        msg = f"Collection '{collection_name}' not found: {exc}"
        raise click.ClickException(msg) from exc

    result = raw_col.get(include=["documents", "metadatas", "ids"])
    raw_docs: list[str | None] = result.get("documents") or []
    raw_meta: list[dict[str, Any] | None] = result.get("metadatas") or []
    raw_ids: list[str] = result.get("ids") or []

    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []

    for text, meta, doc_id in zip(raw_docs, raw_meta, raw_ids, strict=False):
        if not isinstance(text, str) or not text.strip():
            continue
        clean_meta: dict[str, Any] = {}
        if isinstance(meta, dict):
            for key, val in meta.items():
                # Strip old embedding fingerprint keys — new ones are set via the fingerprint
                if key in _FINGERPRINT_KEYS:
                    continue
                clean_meta[key] = val
        texts.append(text)
        metadatas.append(clean_meta)
        ids.append(doc_id)

    logger.debug(f"Fetched {len(texts)} documents from '{collection_name}'")
    return _CollectionDocs(texts=texts, metadatas=metadatas, ids=ids)


# ── Temp Collection Builder ───────────────────────────────────────────────────


def _temp_collection_name(collection_name: str) -> str:
    ts = int(datetime.now(tz=UTC).timestamp())
    return f"{collection_name}_{_MIGRATION_SUFFIX}_{ts}"


def _build_migrated_collection(
    docs: _CollectionDocs,
    embedder: HuggingFaceEmbeddings,
    client: chromadb.PersistentClient,
    temp_name: str,
    fingerprint: dict[str, object],
) -> None:
    """Index *docs* into a new persistent collection called *temp_name*."""
    chroma_db = Chroma(
        client=client,
        collection_name=temp_name,
        embedding_function=embedder,
        collection_metadata=fingerprint,
    )

    for batch_start in range(0, len(docs.texts), _EMBED_BATCH_SIZE):
        batch_texts = docs.texts[batch_start : batch_start + _EMBED_BATCH_SIZE]
        batch_meta = docs.metadatas[batch_start : batch_start + _EMBED_BATCH_SIZE]
        batch_ids = docs.ids[batch_start : batch_start + _EMBED_BATCH_SIZE]
        chroma_db.add_texts(batch_texts, metadatas=batch_meta, ids=batch_ids)
        logger.debug(f"Indexed batch [{batch_start}:{batch_start + len(batch_texts)}] into '{temp_name}'")

    logger.info(f"Built temp collection '{temp_name}' with {len(docs.texts)} documents")


# ── Validation Gate ───────────────────────────────────────────────────────────


def _validate_migrated_collection(
    client: chromadb.PersistentClient,
    temp_name: str,
    target_name: str,
    embedder: HuggingFaceEmbeddings,
    options: _ValidationOptions,
) -> float:
    """Run fixture evaluation on the temp collection and return Recall@k.

    Temporarily renames *temp_name* to *target_name* for evaluation (so the fixture
    collection reference resolves), then renames back. This is safe because the
    original collection has not been deleted yet at this stage.
    """
    # Rename temp → target so fixture queries find the right collection name
    temp_col = client.get_collection(temp_name)
    temp_col.modify(name=target_name)
    logger.debug(f"Temporarily renamed '{temp_name}' → '{target_name}' for validation")

    try:
        eval_options = FixtureEvalOptions(
            fixture_file=options.fixture_file,
            k=None,
            retrieval_mode="similarity",
            persist_directory=options.persist_directory,
            embedding_model=embedder.model_name,
            embedding_device=None,
            show_failures=False,
        )
        run = _execute_fixture_evaluation(eval_options)
        recall = float(run.metrics.get("recall_at_k", 0.0))
        click.echo(f"  Validation Recall@{run.default_k}={recall:.4f} (threshold={options.threshold:.4f})")
    finally:
        # Always rename back so the caller controls cleanup/commit
        target_col = client.get_collection(target_name)
        target_col.modify(name=temp_name)
        logger.debug(f"Renamed '{target_name}' back to '{temp_name}'")

    return recall


# ── Atomic Swap ───────────────────────────────────────────────────────────────


def _commit_migration(
    client: chromadb.PersistentClient,
    source_name: str,
    temp_name: str,
) -> None:
    """Delete *source_name* and rename *temp_name* → *source_name*."""
    client.delete_collection(source_name)
    logger.info(f"Deleted old collection '{source_name}'")
    migrated_col = client.get_collection(temp_name)
    migrated_col.modify(name=source_name)
    logger.info(f"Renamed '{temp_name}' → '{source_name}'")


def _cleanup_temp(client: chromadb.PersistentClient, temp_name: str) -> None:
    """Remove a temp collection left over from a failed or dry-run migration."""
    try:
        client.delete_collection(temp_name)
        logger.debug(f"Cleaned up temp collection '{temp_name}'")
    except Exception:
        logger.warning(f"Could not clean up temp collection '{temp_name}' — may need manual removal")


# ── Single-Collection Migration ───────────────────────────────────────────────


def _read_source_model(client: chromadb.PersistentClient, collection_name: str) -> str:
    """Return the embedding model stored in the collection metadata, or 'unknown'."""
    try:
        col = client.get_collection(collection_name)
        meta = col.metadata or {}
        return str(meta.get(EMBEDDING_MODEL_METADATA_KEY, "unknown"))
    except Exception:
        return "unknown"


def migrate_collection(  # noqa: PLR0911
    spec: MigrationSpec,
    config: MigrationConfig,
) -> MigrationResult:
    """Migrate a single collection to a new embedding model.

    Args:
        spec: Describes source collection name and target model.
        config: Operational config (persist directory, dry-run flag, validation options).
    """
    collection_name = spec.collection_name
    source_model = "unknown"
    temp_name = _temp_collection_name(collection_name)

    client = chromadb.PersistentClient(
        path=config.persist_directory, settings=Settings(anonymized_telemetry=False)
    )
    source_model = _read_source_model(client, collection_name)

    click.echo(f"\n[migrate] {collection_name}")
    click.echo(f"  {source_model} → {spec.target_model_id}")
    if config.dry_run:
        click.secho("  (dry-run: temp collection will be cleaned up, no rename)", fg="yellow")

    docs: _CollectionDocs
    try:
        docs = _fetch_collection_documents(client, collection_name)
    except click.ClickException as exc:
        return MigrationResult(
            collection_name=collection_name,
            source_model=source_model,
            target_model=spec.target_model_id,
            docs_migrated=0,
            status="error",
            error=str(exc),
        )

    if not docs.texts:
        click.secho(f"  WARNING: Collection '{collection_name}' has no documents — skipping", fg="yellow")
        return MigrationResult(
            collection_name=collection_name,
            source_model=source_model,
            target_model=spec.target_model_id,
            docs_migrated=0,
            status="error",
            error="No documents in source collection",
        )

    embedder = HuggingFaceEmbeddings(
        model_name=spec.target_model_id,
        model_kwargs={"device": spec.target_device},
        encode_kwargs={"normalize_embeddings": spec.target_normalize},
        cache_folder=config.embedding_cache,
    )
    fingerprint = build_embedding_fingerprint(
        embedding_model=spec.target_model_id,
        normalize_embeddings=spec.target_normalize,
        embedding_dimension=infer_embedding_dimension(embedder),
    )

    click.echo(f"  Indexing {len(docs.texts)} documents...")
    try:
        _build_migrated_collection(docs, embedder, client, temp_name, fingerprint)
    except Exception as exc:
        _cleanup_temp(client, temp_name)
        return MigrationResult(
            collection_name=collection_name,
            source_model=source_model,
            target_model=spec.target_model_id,
            docs_migrated=0,
            status="error",
            error=f"Failed to build temp collection: {exc}",
        )

    # Validation gate
    validation_recall: float | None = None
    if config.fixture_file is not None:
        click.echo("  Validating...")
        try:
            validation_recall = _validate_migrated_collection(
                client=client,
                temp_name=temp_name,
                target_name=collection_name,
                embedder=embedder,
                options=_ValidationOptions(
                    fixture_file=config.fixture_file,
                    persist_directory=config.persist_directory,
                    threshold=config.validation_threshold,
                ),
            )
        except Exception as exc:
            _cleanup_temp(client, temp_name)
            return MigrationResult(
                collection_name=collection_name,
                source_model=source_model,
                target_model=spec.target_model_id,
                docs_migrated=len(docs.texts),
                status="error",
                error=f"Validation error: {exc}",
                validation_recall=None,
            )

        if validation_recall < config.validation_threshold:
            _cleanup_temp(client, temp_name)
            msg = f"Recall@k={validation_recall:.4f} below threshold={config.validation_threshold:.4f}"
            click.secho(f"  FAIL: {msg}", fg="red")
            return MigrationResult(
                collection_name=collection_name,
                source_model=source_model,
                target_model=spec.target_model_id,
                docs_migrated=len(docs.texts),
                status="validation_failed",
                validation_recall=validation_recall,
                error=msg,
            )

    if config.dry_run:
        _cleanup_temp(client, temp_name)
        click.secho(f"  DRY-RUN OK — {len(docs.texts)} docs would have been migrated", fg="cyan")
        return MigrationResult(
            collection_name=collection_name,
            source_model=source_model,
            target_model=spec.target_model_id,
            docs_migrated=len(docs.texts),
            status="dry_run",
            validation_recall=validation_recall,
        )

    _commit_migration(client, collection_name, temp_name)
    click.secho(f"  OK — migrated {len(docs.texts)} docs", fg="green")

    return MigrationResult(
        collection_name=collection_name,
        source_model=source_model,
        target_model=spec.target_model_id,
        docs_migrated=len(docs.texts),
        status="success",
        validation_recall=validation_recall,
    )


# ── Multi-Collection Orchestration ────────────────────────────────────────────


def run_migration(
    specs: list[MigrationSpec],
    config: MigrationConfig | None = None,
) -> list[MigrationResult]:
    """Migrate a list of collections and return per-collection results.

    Args:
        specs: Migration specifications (collection name + target model).
        config: Operational config. Falls back to app config defaults when None.
    """
    if not specs:
        msg = "specs must contain at least one MigrationSpec"
        raise click.ClickException(msg)

    if config is None:
        app_config = load_app_config()
        script_config = load_rag_script_config(app_config)
        config = MigrationConfig(
            persist_directory=script_config.persist_directory,
            embedding_cache=script_config.embedding_cache,
        )

    results: list[MigrationResult] = []
    for spec in specs:
        result = migrate_collection(spec, config)
        results.append(result)

    return results


# ── Output Formatters ─────────────────────────────────────────────────────────


def format_migration_report(results: list[MigrationResult]) -> str:
    """Return a human-readable summary table of migration results."""
    if not results:
        return "No migrations were run."

    lines = ["Migration Report", "=" * 60]
    for result in results:
        status_icon = "\u2714" if result.succeeded else "\u2718"
        lines.append(f"\n{status_icon} {result.collection_name}  [{result.status.upper()}]")
        lines.append(f"  {result.source_model} → {result.target_model}")
        lines.append(f"  Documents: {result.docs_migrated}")
        if result.validation_recall is not None:
            lines.append(f"  Validation Recall@k: {result.validation_recall:.4f}")
        if result.error:
            lines.append(f"  Error: {result.error}")

    successes = sum(1 for r in results if r.status == "success")
    dry_runs = sum(1 for r in results if r.status == "dry_run")
    failures = len(results) - successes - dry_runs
    lines.append("\n" + "-" * 60)
    lines.append(f"Total: {len(results)}  Success: {successes}  Dry-run: {dry_runs}  Failed: {failures}")
    return "\n".join(lines)


def write_migration_report_json(output_path: Path, results: list[MigrationResult]) -> None:
    """Write migration results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "collection_name": r.collection_name,
            "source_model": r.source_model,
            "target_model": r.target_model,
            "docs_migrated": r.docs_migrated,
            "status": r.status,
            "validation_recall": r.validation_recall,
            "error": r.error,
            "generated_at": r.generated_at,
        }
        for r in results
    ]
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_migration_report_csv(output_path: Path, results: list[MigrationResult]) -> None:
    """Write migration results to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "collection_name",
        "status",
        "source_model",
        "target_model",
        "docs_migrated",
        "validation_recall",
        "error",
        "generated_at",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "collection_name": result.collection_name,
                    "status": result.status,
                    "source_model": result.source_model,
                    "target_model": result.target_model,
                    "docs_migrated": result.docs_migrated,
                    "validation_recall": (
                        f"{result.validation_recall:.6f}" if result.validation_recall is not None else ""
                    ),
                    "error": result.error or "",
                    "generated_at": result.generated_at,
                }
            )
