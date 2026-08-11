"""Thin façade over scripts/rag/* for the RAG Management web UI.

All functions are synchronous and designed to be called from route handlers
via asyncio.to_thread. None of these functions load the LLM model.
"""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chromadb
from chromadb.config import Settings

from core.rag_dependencies import import_vector_dependencies

if TYPE_CHECKING:
    from core.config import RagScriptConfig

_SAFE_STEM_RE: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_-]+$")


def is_valid_stem(stem: str) -> bool:
    """Return True when *stem* contains only letters, digits, underscores, and hyphens."""
    return bool(_SAFE_STEM_RE.match(stem))


def _chroma_client(persist_dir: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


def list_collections(config: RagScriptConfig) -> list[dict[str, Any]]:
    """List all ChromaDB collections with counts and fingerprint metadata."""
    client = _chroma_client(config.persist_directory)
    results: list[dict[str, Any]] = []
    for col in client.list_collections():
        try:
            count = col.count()
        except Exception:
            count = None
        meta = col.metadata or {}
        results.append(
            {
                "name": col.name,
                "count": count,
                "embedding_model": meta.get("embedding:model", ""),
                "embedding_dimension": meta.get("embedding:dimension", ""),
                "embedding_normalize": meta.get("embedding:normalize", ""),
            }
        )
    results.sort(key=lambda c: c["name"])
    return results


def collection_info(config: RagScriptConfig, name: str) -> dict[str, Any] | None:
    """Return detailed info for a single collection, or None if not found."""
    client = _chroma_client(config.persist_directory)
    try:
        col = client.get_collection(name)
    except Exception:
        return None
    try:
        count = col.count()
    except Exception:
        count = 0
    meta = col.metadata or {}
    try:
        sample = col.peek(limit=5)
        sample_docs = [
            {"id": id_, "text": (doc or "")[:200], "metadata": m}
            for id_, doc, m in zip(
                sample.get("ids", []),
                sample.get("documents", []) or [],
                sample.get("metadatas", []) or [],
                strict=False,
            )
        ]
    except Exception:
        sample_docs = []
    return {
        "name": name,
        "count": count,
        "metadata": meta,
        "embedding_model": meta.get("embedding:model", ""),
        "embedding_dimension": meta.get("embedding:dimension", ""),
        "embedding_normalize": meta.get("embedding:normalize", ""),
        "sample_docs": sample_docs,
    }


def delete_collection(config: RagScriptConfig, name: str) -> None:
    """Delete a ChromaDB collection by name."""
    client = _chroma_client(config.persist_directory)
    client.delete_collection(name)


def query_collection(
    config: RagScriptConfig,
    name: str,
    query: str,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Run ad-hoc similarity search. Returns top-k chunks with scores."""
    _chromadb_module, _settings_cls, chroma_cls, huggingface_embeddings_cls = import_vector_dependencies()

    embedder = huggingface_embeddings_cls(
        model_name=config.embedding_model,
        model_kwargs={"device": config.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=config.embedding_cache,
    )
    client = _chroma_client(config.persist_directory)
    db = chroma_cls(
        client=client,
        collection_name=name,
        embedding_function=embedder,
    )
    results = db.similarity_search_with_score(query, k=k)
    return [
        {
            "rank": i + 1,
            "text": doc.page_content,
            "score": round(float(score), 4),
            "metadata": doc.metadata,
        }
        for i, (doc, score) in enumerate(results)
    ]


def backfill_fingerprint(config: RagScriptConfig, name: str) -> dict[str, Any]:
    """Write embedding fingerprint metadata onto an existing collection."""
    _chromadb_module, _settings_cls, _chroma_cls, huggingface_embeddings_cls = import_vector_dependencies()

    from scripts.rag.manage_collections_core_collection import (  # noqa: PLC0415
        build_embedding_fingerprint,
        infer_embedding_dimension,
    )

    embedder = huggingface_embeddings_cls(
        model_name=config.embedding_model,
        model_kwargs={"device": config.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=config.embedding_cache,
    )
    dimension = infer_embedding_dimension(embedder)
    fingerprint = build_embedding_fingerprint(
        embedding_model=config.embedding_model,
        normalize_embeddings=True,
        embedding_dimension=dimension,
    )
    client = _chroma_client(config.persist_directory)
    col = client.get_collection(name)
    existing_meta = col.metadata or {}
    col.modify(metadata={**existing_meta, **fingerprint})
    return fingerprint


# ---------------------------------------------------------------------------
# RAG Data Files
# ---------------------------------------------------------------------------


def list_rag_files(config: RagScriptConfig) -> list[dict[str, Any]]:
    """List .txt source files in rag_data/ with type classification."""
    rag_dir = Path(config.documents_directory)
    if not rag_dir.exists():
        return []
    files: list[dict[str, Any]] = []
    for path in sorted(rag_dir.glob("*.txt")):
        stem = path.stem
        files.append(
            {
                "name": path.name,
                "stem": stem,
                "type": "message_examples" if stem.endswith("_message_examples") else "lore",
                "size": path.stat().st_size,
                "has_metadata": (rag_dir / f"{stem}.json").exists(),
            }
        )
    return files


def file_content(config: RagScriptConfig, filename: str) -> str | None:
    """Return the text content of a rag_data file, guarding against path traversal."""
    rag_dir = Path(config.documents_directory).resolve()
    candidate = (rag_dir / filename).resolve()
    if not candidate.is_relative_to(rag_dir):
        return None
    if not candidate.exists() or not candidate.is_file():
        return None
    if candidate.suffix not in {".txt", ".json"}:
        return None
    return candidate.read_text(encoding="utf-8")


def save_rag_file(config: RagScriptConfig, stem: str, content: bytes) -> dict[str, Any]:
    """Save *content* as ``{stem}.txt`` in the rag_data directory.

    Raises ``ValueError`` if *stem* is invalid.
    Returns a file-info dict matching the shape produced by :func:`list_rag_files`.
    """
    if not is_valid_stem(stem):
        msg = f"Invalid stem {stem!r}: only letters, digits, underscores, and hyphens are allowed."
        raise ValueError(msg)
    rag_dir = Path(config.documents_directory)
    rag_dir.mkdir(parents=True, exist_ok=True)
    dest = rag_dir / f"{stem}.txt"
    dest.write_bytes(content)
    return {
        "name": dest.name,
        "stem": stem,
        "type": "message_examples" if stem.endswith("_message_examples") else "lore",
        "size": len(content),
        "has_metadata": (rag_dir / f"{stem}.json").exists(),
    }


def list_rag_stems(config: RagScriptConfig) -> list[str]:
    """Return a sorted list of stems for all .txt files in rag_data/."""
    rag_dir = Path(config.documents_directory)
    if not rag_dir.exists():
        return []
    return sorted(p.stem for p in rag_dir.glob("*.txt"))


# ---------------------------------------------------------------------------
# Linting
# ---------------------------------------------------------------------------


def run_lint(config: RagScriptConfig, *, auto_fix: bool = False) -> list[dict[str, Any]]:
    """Lint all *_message_examples.txt files. Returns list of report dicts."""
    from scripts.rag.lint_message_examples import lint_file_path  # noqa: PLC0415

    rag_dir = Path(config.documents_directory)
    reports: list[dict[str, Any]] = []
    for path in sorted(rag_dir.glob("*_message_examples.txt")):
        report = lint_file_path(path, auto_fix=auto_fix)
        reports.append(
            {
                "file": path.name,
                "valid": report.valid,
                "auto_fixed": report.auto_fixed,
                "violations": [
                    {
                        "line_no": v.line_no,
                        "rule_id": v.rule_id,
                        "message": v.message,
                        "severity": v.severity.value if hasattr(v.severity, "value") else str(v.severity),
                        "suggested_fix": v.suggested_fix,
                    }
                    for v in report.violations
                ],
            }
        )
    return reports


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def run_coverage(config: RagScriptConfig, stem: str) -> dict[str, Any] | None:
    """Run coverage analysis for a character (lore + metadata pair)."""
    from scripts.rag.analyze_rag_coverage import (  # noqa: PLC0415
        extract_coverage_metrics,
        format_coverage_report,
        load_metadata_file,
    )

    if not is_valid_stem(stem):
        msg = f"Invalid stem {stem!r}: only letters, digits, underscores, and hyphens are allowed"
        raise ValueError(msg)

    rag_dir = Path(config.documents_directory).resolve()
    source_file = (rag_dir / f"{stem}.txt").resolve()
    metadata_file = (rag_dir / f"{stem}.json").resolve()
    if not source_file.is_relative_to(rag_dir) or not metadata_file.is_relative_to(rag_dir):
        msg = f"Stem {stem!r} resolves outside documents directory"
        raise ValueError(msg)
    if not source_file.exists() or not metadata_file.exists():
        return None
    source_text = source_file.read_text(encoding="utf-8")
    metadata_list = load_metadata_file(metadata_file)
    metrics = extract_coverage_metrics(source_text, metadata_list)
    report_text = format_coverage_report(metrics)
    return {
        "stem": stem,
        "entities_count": metrics.entities_count,
        "source_coverage_ratio": round(metrics.source_coverage_ratio, 4),
        "total_source_chars": metrics.total_source_chars,
        "covered_chars": metrics.covered_chars,
        "unmapped_segments": metrics.unmapped_segments[:20],
        "category_distribution": metrics.category_distribution,
        "report_text": report_text,
    }


# ---------------------------------------------------------------------------
# Fixture Evaluation
# ---------------------------------------------------------------------------


def list_fixture_packs(tests_dir: str = "tests/fixtures") -> list[str]:
    """List available fixture JSON files."""
    fixture_dir = Path(tests_dir)
    if not fixture_dir.exists():
        return []
    return sorted(p.name for p in fixture_dir.glob("*.json"))


def run_evaluate_fixtures(
    config: RagScriptConfig,
    fixture_file: str,
    tests_dir: str = "tests/fixtures",
) -> dict[str, Any] | None:
    """Run fixture evaluation in similarity mode. Returns metrics dict."""
    from scripts.rag.manage_collections_core_evaluation import _execute_fixture_evaluation  # noqa: PLC0415
    from scripts.rag.manage_collections_core_types import FixtureEvalOptions  # noqa: PLC0415

    available_fixtures = set(list_fixture_packs(tests_dir))
    if fixture_file not in available_fixtures:
        msg = f"Unknown fixture pack: {fixture_file!r}"
        raise FileNotFoundError(msg)

    fixture_dir = Path(tests_dir).resolve()
    fixture_path = (fixture_dir / fixture_file).resolve()
    if not fixture_path.is_relative_to(fixture_dir) or not fixture_path.exists():
        msg = f"Fixture pack not found: {fixture_file!r}"
        raise FileNotFoundError(msg)
    options = FixtureEvalOptions(
        fixture_file=fixture_path,
        k=None,
        retrieval_mode="similarity",
        persist_directory=config.persist_directory,
        embedding_model=config.embedding_model,
        embedding_device=config.embedding_device,
        show_failures=False,
    )
    run = _execute_fixture_evaluation(options)
    return {
        "fixture_file": fixture_file,
        "default_k": run.default_k,
        "skipped": run.skipped,
        "metrics": run.metrics,
        "case_results": [
            {
                "case_id": c.case_id,
                "rank": c.rank,
                "status": c.status,
                "query": c.query[:120],
                "collection": c.collection,
                "forbidden_hit": c.forbidden_hit,
                "precision_at_k": round(c.precision_at_k, 4),
                "average_precision_at_k": round(c.average_precision_at_k, 4),
                "matched_expected": c.matched_expected,
                "expected_total": c.expected_total,
            }
            for c in run.case_results
        ],
    }


def get_fixture_trends(logs_dir: str = "logs/retrieval_eval") -> list[dict[str, Any]]:
    """Read retrieval evaluation trend history from CSV (newest first)."""
    history_path = Path(logs_dir) / "history.csv"
    if not history_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with history_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows.extend(dict(row) for row in reader)
    except Exception:
        return []
    return list(reversed(rows))


# ---------------------------------------------------------------------------
# Collection Push
# ---------------------------------------------------------------------------


def push_collection(
    config: RagScriptConfig,
    stem: str,
    collection_name: str,
    *,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Chunk, enrich, and push a rag_data text file into a ChromaDB collection."""
    _chromadb_module, _settings_cls, _chroma_cls, huggingface_embeddings_cls = import_vector_dependencies()

    from scripts.rag.push_rag_data import (  # noqa: PLC0415
        ProcessingContext,
        PushConfig,
        build_embedding_fingerprint,
        enrich_documents_with_metadata,
        infer_embedding_dimension,
        load_and_chunk_text_file,
        push_to_collection,
        resolve_metadata_file,
    )

    rag_dir = Path(config.documents_directory).resolve()
    if not is_valid_stem(stem):
        msg = f"Invalid stem {stem!r}: only letters, digits, underscores, and hyphens are allowed"
        raise ValueError(msg)
    file_path = (rag_dir / f"{stem}.txt").resolve()
    if not file_path.is_relative_to(rag_dir):
        msg = f"Stem {stem!r} resolves outside documents directory"
        raise ValueError(msg)
    if not file_path.exists():
        msg = f"Source file not found: {file_path}"
        raise FileNotFoundError(msg)

    embedder = huggingface_embeddings_cls(
        model_name=config.embedding_model,
        model_kwargs={"device": config.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=config.embedding_cache,
    )
    client = _chroma_client(config.persist_directory)
    dimension = infer_embedding_dimension(embedder)
    fingerprint = build_embedding_fingerprint(
        embedding_model=config.embedding_model,
        normalize_embeddings=True,
        embedding_dimension=dimension,
    )
    documents = load_and_chunk_text_file(file_path, config.chunk_size, config.chunk_overlap)
    metadata_file = resolve_metadata_file(file_path, config.key_storage, None)
    documents = enrich_documents_with_metadata(documents, metadata_file, config.threads)

    push_cfg = PushConfig(
        persist_directory=config.persist_directory,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        key_storage=config.key_storage,
        threads=config.threads,
        dry_run=False,
        overwrite=overwrite,
    )
    ctx = ProcessingContext(embedder=embedder, client=client)
    t0 = time.monotonic()
    push_to_collection(collection_name, documents, push_cfg, ctx, fingerprint)
    elapsed = time.monotonic() - t0
    return {
        "collection": collection_name,
        "stem": stem,
        "doc_count": len(documents),
        "elapsed_s": round(elapsed, 2),
    }


def get_benchmark_results(benchmark_dir: str = "logs/benchmark") -> dict[str, Any] | None:
    """Load the most recent benchmark JSON from logs/benchmark/, if present."""
    benchmark_path = Path(benchmark_dir) / "last_benchmark.json"
    if not benchmark_path.exists():
        return None
    try:
        with benchmark_path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
