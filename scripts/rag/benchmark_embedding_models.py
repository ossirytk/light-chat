"""Embedding model benchmark harness.

Compares multiple embedding models by building fresh in-memory reproductions of existing
collections, then running retrieval fixture cases against each to measure Recall@k, MRR, and
MAP@k side by side. Results are printed as a ranked comparison table.

Benchmark configuration JSON format::

    {
        "models": [
            {"model_id": "sentence-transformers/all-mpnet-base-v2", "normalize": true,
             "device": "cpu", "label": "mpnet-base-v2"},
            {"model_id": "sentence-transformers/all-MiniLM-L6-v2", "normalize": true,
             "device": "cpu", "label": "MiniLM-L6-v2"}
        ]
    }

A bare list ``[{"model_id": ..., ...}]`` is also accepted.
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
from scripts.rag.manage_collections_core_evaluation import _load_fixture_payload
from scripts.rag.manage_collections_core_metrics import compute_case_match_details, compute_run_metrics
from scripts.rag.manage_collections_core_types import FixtureCaseResult

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_FIXTURE = Path("tests/fixtures/retrieval_fixtures.json")
_TABLE_COL_MODEL = 35
_TABLE_COL_METRIC = 9
_EMBED_BATCH_SIZE = 256
_BENCHMARK_VERSION = "1"


# ── Data Types ────────────────────────────────────────────────────────────────


@dataclass
class EmbeddingModelSpec:
    """Specification for a single embedding model to benchmark."""

    model_id: str
    normalize: bool = True
    device: str = "cpu"
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.model_id.split("/")[-1]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "normalize": self.normalize,
            "device": self.device,
            "label": self.label,
        }


@dataclass
class BenchmarkModelResult:
    """Evaluation metrics and per-case results for one model pass."""

    label: str
    model_id: str
    metrics: dict[str, float]
    case_results: list[FixtureCaseResult]
    evaluated: int
    skipped: int


@dataclass
class BenchmarkRun:
    """All model results from a single benchmark execution."""

    fixture_file: Path
    k: int
    dashboard_ks: list[int]
    model_results: list[BenchmarkModelResult]
    generated_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())


# ── Config Loading ────────────────────────────────────────────────────────────


def load_model_specs_from_json(config_path: Path) -> list[EmbeddingModelSpec]:
    """Load a list of EmbeddingModelSpec from a JSON file.

    Accepts a bare list or a dict with a ``"models"`` key.
    """
    if not config_path.exists():
        msg = f"Benchmark config not found: {config_path}"
        raise FileNotFoundError(msg)

    raw: object = json.loads(config_path.read_text(encoding="utf-8"))

    if isinstance(raw, dict) and "models" in raw:
        raw = raw["models"]

    if not isinstance(raw, list):
        msg = f"Expected a list of model specs (or dict with 'models' key), got {type(raw).__name__}"
        raise TypeError(msg)

    specs: list[EmbeddingModelSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("model_id", ""))
        if not model_id:
            continue
        specs.append(
            EmbeddingModelSpec(
                model_id=model_id,
                normalize=bool(item.get("normalize", True)),
                device=str(item.get("device", "cpu")),
                label=str(item.get("label", "")),
            )
        )
    return specs


# ── Corpus Extraction ─────────────────────────────────────────────────────────


def _fetch_corpus_texts(
    client: chromadb.PersistentClient,
    collection_names: set[str],
) -> dict[str, list[str]]:
    """Return mapping of collection_name -> list[text] from the persisted client."""
    available = {col.name for col in client.list_collections()}
    corpus: dict[str, list[str]] = {}
    for name in collection_names:
        if name not in available:
            logger.warning(f"Collection '{name}' not found in persistent store — fixture cases will be skipped")
            corpus[name] = []
            continue

        raw_col = client.get_collection(name)
        result = raw_col.get(include=["documents"])
        raw_docs: list[str | None] = result.get("documents") or []
        texts = [doc for doc in raw_docs if isinstance(doc, str) and doc.strip()]
        logger.debug(f"Fetched {len(texts)} documents from collection '{name}'")
        corpus[name] = texts

    return corpus


# ── Ephemeral Collection Builder ──────────────────────────────────────────────


def _build_ephemeral_collection(
    texts: list[str],
    embedder: HuggingFaceEmbeddings,
    collection_name: str,
) -> Chroma:
    """Create an in-memory Chroma collection indexed with *embedder*."""
    ephemeral_client = chromadb.EphemeralClient()
    chroma_db = Chroma(
        client=ephemeral_client,
        collection_name=collection_name,
        embedding_function=embedder,
    )
    for batch_start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[batch_start : batch_start + _EMBED_BATCH_SIZE]
        chroma_db.add_texts(batch)

    logger.debug(f"Indexed {len(texts)} texts into ephemeral '{collection_name}'")
    return chroma_db


# ── Per-Case Evaluation ───────────────────────────────────────────────────────


def _evaluate_single_case(
    case: dict[str, Any],
    default_collection: str,
    db_cache: dict[str, Chroma],
    default_k: int,
    evaluation_k: int,
) -> FixtureCaseResult:
    """Evaluate one fixture case against the ephemeral db_cache."""
    case_id = str(case.get("id", "unknown_case"))
    collection_name = str(case.get("collection", default_collection))
    query = str(case.get("query", ""))

    expected_snippets_raw = case.get("expected_snippets", [])
    forbidden_snippets_raw = case.get("forbidden_snippets", [])
    min_expected_matches_raw = case.get("min_expected_matches", 1)

    expected_snippets = [s for s in expected_snippets_raw if isinstance(s, str)]
    forbidden_snippets = [s for s in forbidden_snippets_raw if isinstance(s, str)]
    min_expected_matches = int(min_expected_matches_raw) if isinstance(min_expected_matches_raw, int) else 1

    if not query or not isinstance(expected_snippets_raw, list):
        return FixtureCaseResult(
            case_id=case_id,
            rank=None,
            status="invalid",
            query=query,
            collection=collection_name,
            expected_snippets=expected_snippets,
            min_expected_matches=max(1, min_expected_matches),
            forbidden_snippets=forbidden_snippets,
        )

    if collection_name not in db_cache:
        return FixtureCaseResult(
            case_id=case_id,
            rank=None,
            status="missing_collection",
            query=query,
            collection=collection_name,
            expected_snippets=expected_snippets,
            min_expected_matches=min_expected_matches,
            forbidden_snippets=forbidden_snippets,
        )

    docs = db_cache[collection_name].similarity_search_with_score(query=query, k=evaluation_k)
    chunks = [doc.page_content for doc, _score in docs]

    details = compute_case_match_details(
        chunks=chunks,
        expected_snippets=expected_snippets,
        forbidden_snippets=forbidden_snippets,
        k=default_k,
    )
    match_ranks: dict[str, int] = details["match_ranks"]  # type: ignore[assignment]
    expected_total = int(details["expected_total"])  # type: ignore[arg-type]
    matched_expected = int(details["matched_expected"])  # type: ignore[arg-type]
    rank = min(match_ranks.values()) if matched_expected >= min_expected_matches and match_ranks else None
    forbidden_matches = list(details["forbidden_matches"])  # type: ignore[arg-type]
    forbidden_hit = len(forbidden_matches) > 0

    return FixtureCaseResult(
        case_id=case_id,
        rank=rank,
        status="ok",
        query=query,
        collection=collection_name,
        expected_snippets=expected_snippets,
        min_expected_matches=min_expected_matches,
        expected_total=expected_total,
        matched_expected=matched_expected,
        expected_recall_at_k=float(details["expected_recall_at_k"]),  # type: ignore[arg-type]
        precision_at_k=float(details["precision_at_k"]),  # type: ignore[arg-type]
        average_precision_at_k=float(details["average_precision_at_k"]),  # type: ignore[arg-type]
        forbidden_snippets=forbidden_snippets,
        forbidden_matches=forbidden_matches,
        forbidden_hit=forbidden_hit,
    )


# ── Per-Model Run ─────────────────────────────────────────────────────────────


@dataclass
class _ModelRunContext:
    """Internal context bundle passed to each per-model evaluation pass."""

    corpus_by_collection: dict[str, list[str]]
    cases: list[dict[str, Any]]
    default_collection: str
    k: int
    evaluation_k: int
    dashboard_ks: list[int]
    embedding_cache: str


def _run_model_benchmark(
    spec: EmbeddingModelSpec,
    ctx: _ModelRunContext,
) -> BenchmarkModelResult:
    """Embed the corpus with *spec*, run all fixture cases, and return results."""
    click.echo(f"\n[{spec.label}] {spec.model_id}")
    embedder = HuggingFaceEmbeddings(
        model_name=spec.model_id,
        model_kwargs={"device": spec.device},
        encode_kwargs={"normalize_embeddings": spec.normalize},
        cache_folder=ctx.embedding_cache,
    )

    db_cache: dict[str, Chroma] = {
        name: _build_ephemeral_collection(texts, embedder, name)
        for name, texts in ctx.corpus_by_collection.items()
        if texts
    }

    case_results: list[FixtureCaseResult] = []
    skipped = 0
    for case in ctx.cases:
        result = _evaluate_single_case(case, ctx.default_collection, db_cache, ctx.k, ctx.evaluation_k)
        if result.status != "ok":
            skipped += 1
        case_results.append(result)

    metrics = compute_run_metrics(case_results, ctx.dashboard_ks, ctx.k)
    evaluated = len([r for r in case_results if r.status == "ok"])

    recall = metrics.get("recall_at_k", 0.0)
    mrr = metrics.get("mrr", 0.0)
    map_k = metrics.get("map_at_k", 0.0)
    click.echo(f"  Recall@{ctx.k}={recall:.4f}  MRR={mrr:.4f}  MAP@{ctx.k}={map_k:.4f}  [{evaluated} cases]")

    return BenchmarkModelResult(
        label=spec.label,
        model_id=spec.model_id,
        metrics=metrics,
        case_results=case_results,
        evaluated=evaluated,
        skipped=skipped,
    )


# ── Benchmark Orchestration ───────────────────────────────────────────────────


def run_embedding_benchmark(
    model_specs: list[EmbeddingModelSpec],
    fixture_file: Path,
    persist_directory: str | None = None,
    k_override: int | None = None,
) -> BenchmarkRun:
    """Run a full embedding model comparison benchmark.

    Args:
        model_specs: Models to compare; at least one required.
        fixture_file: Retrieval fixture JSON to use for evaluation.
        persist_directory: Path to persistent ChromaDB. Falls back to app config.
        k_override: Override top-k for all fixture queries.
    """
    if not model_specs:
        msg = "model_specs must contain at least one model"
        raise click.ClickException(msg)

    if not fixture_file.exists():
        msg = f"Fixture file not found: {fixture_file}"
        raise click.ClickException(msg)

    default_collection, fixture_k, dashboard_ks, cases = _load_fixture_payload(fixture_file)
    k = int(k_override if k_override is not None else fixture_k)
    evaluation_k = max([k, *dashboard_ks])

    if not cases:
        msg = "Fixture file has no cases to evaluate"
        raise click.ClickException(msg)

    app_config = load_app_config()
    script_config = load_rag_script_config(app_config)
    resolved_persist = persist_directory or script_config.persist_directory
    embedding_cache = script_config.embedding_cache

    # Collect all collection names referenced in fixture cases
    referenced_collections: set[str] = {default_collection}
    for case in cases:
        if isinstance(case.get("collection"), str):
            referenced_collections.add(case["collection"])

    client = chromadb.PersistentClient(path=resolved_persist, settings=Settings(anonymized_telemetry=False))
    corpus_by_collection = _fetch_corpus_texts(client, referenced_collections)

    click.echo(
        f"\nEmbedding Model Benchmark — {fixture_file} (k={k})"
        f"\nCorpus: {sum(len(v) for v in corpus_by_collection.values())} docs"
        f" across {len(corpus_by_collection)} collection(s)"
        f"\nModels: {len(model_specs)}"
    )

    run_ctx = _ModelRunContext(
        corpus_by_collection=corpus_by_collection,
        cases=cases,
        default_collection=default_collection,
        k=k,
        evaluation_k=evaluation_k,
        dashboard_ks=dashboard_ks,
        embedding_cache=embedding_cache,
    )

    model_results: list[BenchmarkModelResult] = []
    for spec in model_specs:
        result = _run_model_benchmark(spec, run_ctx)
        model_results.append(result)

    return BenchmarkRun(
        fixture_file=fixture_file,
        k=k,
        dashboard_ks=dashboard_ks,
        model_results=model_results,
    )


# ── Output Formatters ─────────────────────────────────────────────────────────


def format_benchmark_table(run: BenchmarkRun) -> str:
    """Format a ranked comparison table for all models in the benchmark run."""
    k = run.k
    sep = "-" * (_TABLE_COL_MODEL + 1 + (_TABLE_COL_METRIC + 1) * 5)
    header = (
        f"{'Model':<{_TABLE_COL_MODEL}} "
        f"{'Recall@' + str(k):>{_TABLE_COL_METRIC}} "
        f"{'MRR':>{_TABLE_COL_METRIC}} "
        f"{'MAP@' + str(k):>{_TABLE_COL_METRIC}} "
        f"{'Cases':>{_TABLE_COL_METRIC}} "
        f"{'Hits':>{_TABLE_COL_METRIC}}"
    )

    def _row(result: BenchmarkModelResult) -> str:
        m = result.metrics
        return (
            f"{result.label:<{_TABLE_COL_MODEL}} "
            f"{m.get('recall_at_k', 0.0):>{_TABLE_COL_METRIC}.4f} "
            f"{m.get('mrr', 0.0):>{_TABLE_COL_METRIC}.4f} "
            f"{m.get('map_at_k', 0.0):>{_TABLE_COL_METRIC}.4f} "
            f"{result.evaluated:>{_TABLE_COL_METRIC}} "
            f"{int(m.get('hits', 0)):>{_TABLE_COL_METRIC}}"
        )

    ranked = sorted(run.model_results, key=lambda r: r.metrics.get("recall_at_k", 0.0), reverse=True)
    rows = [header, sep] + [_row(r) for r in ranked]
    return "\n".join(rows)


def write_benchmark_report_json(output_path: Path, run: BenchmarkRun) -> None:
    """Write full benchmark results as a JSON report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": run.generated_at,
        "benchmark_version": _BENCHMARK_VERSION,
        "fixture_file": str(run.fixture_file),
        "k": run.k,
        "dashboard_ks": run.dashboard_ks,
        "models": [
            {
                "label": r.label,
                "model_id": r.model_id,
                "evaluated": r.evaluated,
                "skipped": r.skipped,
                "metrics": {key: round(val, 6) for key, val in r.metrics.items()},
            }
            for r in run.model_results
        ],
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def write_benchmark_report_csv(output_path: Path, run: BenchmarkRun) -> None:
    """Write per-model benchmark metrics as a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "model_id",
        "evaluated",
        "skipped",
        "recall_at_k",
        "mrr",
        "map_at_k",
        "expected_recall_at_k",
        "precision_at_k",
        "fixture_file",
        "k",
        "generated_at",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in run.model_results:
            m = result.metrics
            writer.writerow(
                {
                    "label": result.label,
                    "model_id": result.model_id,
                    "evaluated": result.evaluated,
                    "skipped": result.skipped,
                    "recall_at_k": f"{m.get('recall_at_k', 0.0):.6f}",
                    "mrr": f"{m.get('mrr', 0.0):.6f}",
                    "map_at_k": f"{m.get('map_at_k', 0.0):.6f}",
                    "expected_recall_at_k": f"{m.get('expected_recall_at_k', 0.0):.6f}",
                    "precision_at_k": f"{m.get('precision_at_k', 0.0):.6f}",
                    "fixture_file": str(run.fixture_file),
                    "k": run.k,
                    "generated_at": run.generated_at,
                }
            )
