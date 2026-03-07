"""Enhanced collection management script for ChromaDB.

This script provides comprehensive collection management:
- List all collections with statistics
- Delete single or multiple collections
- Test collections with similarity search
- Export collection data
- Collection backup and restore
- Bulk operations
"""

import csv
import fnmatch
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import chromadb
import click
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import load_app_config, load_rag_script_config, load_runtime_config
from core.conversation_manager import ConversationManager

type KeyItem = dict[str, object]
type KeyMatch = dict[str, str]
type WhereFilter = dict[str, object] | None


@dataclass
class ManagementContext:
    client: chromadb.PersistentClient
    persist_directory: str
    embedder: HuggingFaceEmbeddings
    key_storage: str


@dataclass
class FixtureEvalContext:
    default_collection: str
    default_k: int
    evaluation_k: int
    dashboard_ks: list[int]
    available_collections: set[str]
    db_cache: dict[str, Chroma]
    retrieval_mode: str
    runtime_manager: object | None
    show_failures: bool


@dataclass
class FixtureCaseResult:
    case_id: str
    rank: int | None
    status: str
    query: str
    collection: str
    expected_snippets: list[str]
    min_expected_matches: int = 1
    expected_total: int = 0
    matched_expected: int = 0
    expected_recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    average_precision_at_k: float = 0.0
    forbidden_snippets: list[str] = field(default_factory=list)
    forbidden_matches: list[str] = field(default_factory=list)
    forbidden_hit: bool = False


@dataclass
class FixtureEvalOptions:
    fixture_file: Path
    k: int | None
    retrieval_mode: str
    persist_directory: str | None
    show_failures: bool


@dataclass
class FixtureEvalRun:
    default_k: int
    case_results: list[FixtureCaseResult]
    skipped: int
    metrics: dict[str, float]
    report: dict[str, Any]


def normalize_keyfile(raw_keys: object) -> list[KeyItem]:
    if isinstance(raw_keys, dict) and "Content" in raw_keys:
        raw_keys = raw_keys["Content"]
    if not isinstance(raw_keys, list):
        return []
    return [item for item in raw_keys if isinstance(item, dict)]


def _get_entry_value(item: KeyItem) -> str | None:
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


def _matches_aliases(item: KeyItem, text_lower: str) -> bool:
    aliases = item.get("aliases")
    if not isinstance(aliases, list):
        return False
    return any(isinstance(alias, str) and alias.lower() in text_lower for alias in aliases)


def extract_key_matches(keys: list[KeyItem], text: str) -> list[KeyMatch]:
    if not text:
        return []
    text_lower = text.lower()
    matches: list[KeyMatch] = []
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


def build_where_filters(matches: list[KeyMatch]) -> list[WhereFilter]:
    if not matches:
        return [None]
    if len(matches) == 1:
        return [matches[0]]
    return [{"$and": matches}, {"$or": matches}]


def get_collection_info(client: chromadb.PersistentClient, collection_name: str) -> dict:
    """Get detailed information about a collection."""
    try:
        collection = client.get_collection(collection_name)
        count = collection.count()
        metadata = collection.metadata

    except ValueError:
        return {
            "name": collection_name,
            "exists": False,
        }
    else:
        return {
            "name": collection_name,
            "count": count,
            "metadata": metadata,
            "exists": True,
        }


def _clean_expected_snippets(expected_snippets: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for snippet in expected_snippets:
        if not isinstance(snippet, str):
            continue
        normalized = snippet.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
    return cleaned


def _first_match_rank(results: list[str], expected_snippets: list[str]) -> int | None:
    cleaned_snippets = _clean_expected_snippets(expected_snippets)
    if not cleaned_snippets:
        return None
    for index, chunk in enumerate(results, start=1):
        if any(snippet in chunk for snippet in cleaned_snippets):
            return index
    return None


def _expected_match_ranks(results: list[str], expected_snippets: list[str], k: int) -> dict[str, int]:
    cleaned_snippets = _clean_expected_snippets(expected_snippets)
    if not cleaned_snippets or k <= 0:
        return {}

    ranks: dict[str, int] = {}
    for index, chunk in enumerate(results[:k], start=1):
        for snippet in cleaned_snippets:
            if snippet in ranks:
                continue
            if snippet in chunk:
                ranks[snippet] = index
    return ranks


def _average_precision_at_k(match_ranks: dict[str, int], expected_total: int, k: int) -> float:
    if expected_total <= 0 or not match_ranks or k <= 0:
        return 0.0

    sorted_ranks = sorted({rank for rank in match_ranks.values() if rank <= k})
    if not sorted_ranks:
        return 0.0

    precision_sum = 0.0
    for hit_count, rank in enumerate(sorted_ranks, start=1):
        precision_sum += min(1.0, hit_count / rank)
    return precision_sum / expected_total


def _find_forbidden_matches(results: list[str], forbidden_snippets: list[str]) -> list[str]:
    cleaned_snippets = _clean_expected_snippets(forbidden_snippets)
    if not cleaned_snippets:
        return []

    merged_results = "\n".join(results)
    return [snippet for snippet in cleaned_snippets if snippet in merged_results]


def _compute_fixture_metrics(ranks: list[int | None], k: int) -> dict[str, float]:
    if not ranks:
        return {
            "cases": 0.0,
            "hits": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
        }

    case_count = len(ranks)
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    mrr_total = sum((1 / rank) if rank is not None and rank > 0 else 0.0 for rank in ranks)

    return {
        "cases": float(case_count),
        "hits": float(hits),
        "recall_at_k": hits / case_count,
        "mrr": mrr_total / case_count,
    }


def _compute_collection_metrics(case_results: list[FixtureCaseResult], k: int) -> dict[str, dict[str, float]]:
    by_collection: dict[str, list[FixtureCaseResult]] = {}
    for result in case_results:
        if result.status != "ok":
            continue
        by_collection.setdefault(result.collection, []).append(result)

    summary: dict[str, dict[str, float]] = {}
    for collection_name, collection_results in by_collection.items():
        ranks = [None if result.forbidden_hit else result.rank for result in collection_results]
        base_metrics = _compute_fixture_metrics(ranks, k)
        macro_expected_recall = sum(result.expected_recall_at_k for result in collection_results) / len(
            collection_results
        )
        macro_precision = sum(result.precision_at_k for result in collection_results) / len(collection_results)
        map_at_k = sum(result.average_precision_at_k for result in collection_results) / len(collection_results)
        summary[collection_name] = {
            **base_metrics,
            "expected_recall_at_k": macro_expected_recall,
            "precision_at_k": macro_precision,
            "map_at_k": map_at_k,
        }
    return summary


def _compute_hit_rate_by_k(ranks: list[int | None], dashboard_ks: list[int]) -> dict[str, float]:
    if not ranks:
        return {f"hit_rate_at_{k}": 0.0 for k in dashboard_ks}

    total = len(ranks)
    return {
        f"hit_rate_at_{k}": sum(1 for rank in ranks if rank is not None and rank <= k) / total for k in dashboard_ks
    }


def _load_fixture_payload(fixture_file: Path) -> tuple[str, int, list[int], list[dict[str, Any]]]:
    fixture_data: dict[str, Any] = json.loads(fixture_file.read_text(encoding="utf-8"))
    default_collection = str(fixture_data.get("collection", ""))
    default_k = int(fixture_data.get("k", 5))
    raw_dashboard_ks = fixture_data.get("dashboard_ks", [1, 3, default_k])
    dashboard_ks = sorted({int(value) for value in raw_dashboard_ks if isinstance(value, int) and value > 0})
    if not dashboard_ks:
        dashboard_ks = [1, default_k]
    raw_cases = fixture_data.get("cases", [])
    if not isinstance(raw_cases, list):
        return default_collection, default_k, dashboard_ks, []
    cases = [case for case in raw_cases if isinstance(case, dict)]
    return default_collection, default_k, dashboard_ks, cases


def _evaluate_fixture_case(
    case: dict[str, Any],
    eval_context: FixtureEvalContext,
) -> FixtureCaseResult:
    case_id = str(case.get("id", "unknown_case"))
    collection_name = str(case.get("collection", eval_context.default_collection))
    query = str(case.get("query", ""))
    expected_snippets_raw = case.get("expected_snippets", [])
    forbidden_snippets_raw = case.get("forbidden_snippets", [])
    min_expected_matches_raw = case.get("min_expected_matches", 1)
    expected_snippets = [snippet for snippet in expected_snippets_raw if isinstance(snippet, str)]
    forbidden_snippets = [snippet for snippet in forbidden_snippets_raw if isinstance(snippet, str)]
    min_expected_matches = int(min_expected_matches_raw) if isinstance(min_expected_matches_raw, int) else -1

    if not collection_name or not query or not isinstance(expected_snippets_raw, list):
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

    if min_expected_matches < 1:
        return FixtureCaseResult(
            case_id=case_id,
            rank=None,
            status="invalid",
            query=query,
            collection=collection_name,
            expected_snippets=expected_snippets,
            min_expected_matches=1,
            forbidden_snippets=forbidden_snippets,
        )

    if forbidden_snippets_raw and not isinstance(forbidden_snippets_raw, list):
        return FixtureCaseResult(
            case_id=case_id,
            rank=None,
            status="invalid",
            query=query,
            collection=collection_name,
            expected_snippets=expected_snippets,
            min_expected_matches=min_expected_matches,
            forbidden_snippets=forbidden_snippets,
        )

    if collection_name not in eval_context.available_collections:
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

    chunks: list[str]
    if eval_context.retrieval_mode == "runtime":
        if eval_context.runtime_manager is None:
            return FixtureCaseResult(
                case_id=case_id,
                rank=None,
                status="invalid",
                query=query,
                collection=collection_name,
                expected_snippets=expected_snippets,
                forbidden_snippets=forbidden_snippets,
            )
        chunks = eval_context.runtime_manager._search_collection(  # noqa: SLF001
            collection_name,
            query,
            [None],
            k=eval_context.evaluation_k,
        )
    else:
        docs = eval_context.db_cache[collection_name].similarity_search_with_score(
            query=query,
            k=eval_context.evaluation_k,
        )
        chunks = [doc.page_content for doc, _score in docs]

    match_ranks = _expected_match_ranks(chunks, expected_snippets, eval_context.default_k)
    expected_total = len(_clean_expected_snippets(expected_snippets))
    matched_expected = len(match_ranks)
    rank = min(match_ranks.values()) if matched_expected >= min_expected_matches and match_ranks else None
    forbidden_matches = _find_forbidden_matches(chunks, forbidden_snippets)
    forbidden_hit = len(forbidden_matches) > 0
    expected_recall_at_k = (matched_expected / expected_total) if expected_total > 0 else 0.0
    matched_ranks_count = len({rank for rank in match_ranks.values() if rank <= eval_context.default_k})
    precision_at_k = (matched_ranks_count / eval_context.default_k) if eval_context.default_k > 0 else 0.0
    average_precision_at_k = _average_precision_at_k(match_ranks, expected_total, eval_context.default_k)
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
        expected_recall_at_k=expected_recall_at_k,
        precision_at_k=precision_at_k,
        average_precision_at_k=average_precision_at_k,
        forbidden_snippets=forbidden_snippets,
        forbidden_matches=forbidden_matches,
        forbidden_hit=forbidden_hit,
    )


def _run_fixture_evaluation(
    cases: list[dict[str, Any]],
    eval_context: FixtureEvalContext,
) -> tuple[list[FixtureCaseResult], int]:
    case_results: list[FixtureCaseResult] = []
    skipped = 0

    click.echo(f"Evaluating {len(cases)} fixture case(s) with k={eval_context.default_k}")
    for case in cases:
        case_result = _evaluate_fixture_case(case, eval_context)
        case_id = case_result.case_id
        rank = case_result.rank
        status = case_result.status

        if status == "invalid":
            skipped += 1
            if eval_context.show_failures:
                click.secho(f"- {case_id}: skipped (invalid fixture fields)", fg="yellow")
            case_results.append(case_result)
            continue

        if status == "missing_collection":
            skipped += 1
            if eval_context.show_failures:
                missing_collection = case_result.collection
                click.secho(f"- {case_id}: skipped (missing collection '{missing_collection}')", fg="yellow")
            case_results.append(case_result)
            continue

        case_results.append(case_result)

        if case_result.forbidden_hit:
            status_text = "FORBIDDEN_HIT"
            color = "red"
        elif rank is None or rank > eval_context.default_k:
            status_text = "MISS"
            color = "red"
        else:
            status_text = f"HIT@{rank}"
            color = "green"
        click.secho(f"- {case_id}: {status_text}", fg=color)

        if (rank is None or case_result.forbidden_hit) and eval_context.show_failures:
            click.echo(f"  query: {case_result.query}")
            click.echo(f"  expected_snippets: {case_result.expected_snippets}")
            if case_result.forbidden_snippets:
                click.echo(f"  forbidden_snippets: {case_result.forbidden_snippets}")
            if case_result.forbidden_matches:
                click.echo(f"  forbidden_matches: {case_result.forbidden_matches}")
            click.echo(
                "  expected_matches: "
                f"{case_result.matched_expected}/{case_result.expected_total} "
                f"(min_required={case_result.min_expected_matches})"
            )

    return case_results, skipped


def _build_fixture_report(
    fixture_file: Path,
    eval_context: FixtureEvalContext,
    metrics: dict[str, float],
    skipped: int,
    case_results: list[FixtureCaseResult],
) -> dict[str, Any]:
    evaluated_cases = [result for result in case_results if result.status == "ok"]
    hit_rate_by_k = _compute_hit_rate_by_k(
        [None if result.forbidden_hit else result.rank for result in evaluated_cases],
        eval_context.dashboard_ks,
    )
    collection_metrics = _compute_collection_metrics(evaluated_cases, eval_context.default_k)
    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "fixture_file": str(fixture_file),
        "k": eval_context.default_k,
        "dashboard_ks": eval_context.dashboard_ks,
        "retrieval_mode": eval_context.retrieval_mode,
        "summary": {
            "evaluated": int(metrics["cases"]),
            "skipped": skipped,
            "hits": int(metrics["hits"]),
            "recall_at_k": metrics["recall_at_k"],
            "mrr": metrics["mrr"],
            "expected_recall_at_k": float(metrics.get("expected_recall_at_k", 0.0)),
            "precision_at_k": float(metrics.get("precision_at_k", 0.0)),
            "map_at_k": float(metrics.get("map_at_k", 0.0)),
            **hit_rate_by_k,
        },
        "collections": collection_metrics,
        "cases": [
            {
                "id": result.case_id,
                "status": "forbidden_match" if result.forbidden_hit else ("hit" if result.rank is not None else "miss"),
                "rank": result.rank,
                "query": result.query,
                "collection": result.collection,
                "expected_snippets": result.expected_snippets,
                "min_expected_matches": result.min_expected_matches,
                "expected_total": result.expected_total,
                "matched_expected": result.matched_expected,
                "expected_recall_at_k": result.expected_recall_at_k,
                "precision_at_k": result.precision_at_k,
                "average_precision_at_k": result.average_precision_at_k,
                "forbidden_snippets": result.forbidden_snippets,
                "forbidden_matches": result.forbidden_matches,
            }
            for result in evaluated_cases
        ],
        "skipped_cases": [
            {
                "id": result.case_id,
                "status": result.status,
                "query": result.query,
                "collection": result.collection,
            }
            for result in case_results
            if result.status != "ok"
        ],
    }


def _write_fixture_report_json(output_file: Path, report: dict[str, Any]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_fixture_report_csv(output_file: Path, case_results: list[FixtureCaseResult], default_k: int) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "id",
                "status",
                "rank",
                "hit_at_k",
                "forbidden_hit",
                "k",
                "collection",
                "query",
                "min_expected_matches",
                "expected_total",
                "matched_expected",
                "expected_recall_at_k",
                "precision_at_k",
                "average_precision_at_k",
                "expected_snippets",
                "forbidden_snippets",
                "forbidden_matches",
            ],
        )
        writer.writeheader()
        for result in case_results:
            hit_at_k = bool(result.rank is not None and result.rank <= default_k and not result.forbidden_hit)
            if result.status != "ok":
                row_status = f"skipped:{result.status}"
            elif result.forbidden_hit:
                row_status = "forbidden_match"
            elif result.rank is None:
                row_status = "miss"
            else:
                row_status = "hit"
            writer.writerow(
                {
                    "id": result.case_id,
                    "status": row_status,
                    "rank": result.rank,
                    "hit_at_k": hit_at_k,
                    "forbidden_hit": result.forbidden_hit,
                    "k": default_k,
                    "collection": result.collection,
                    "query": result.query,
                    "min_expected_matches": result.min_expected_matches,
                    "expected_total": result.expected_total,
                    "matched_expected": result.matched_expected,
                    "expected_recall_at_k": f"{result.expected_recall_at_k:.6f}",
                    "precision_at_k": f"{result.precision_at_k:.6f}",
                    "average_precision_at_k": f"{result.average_precision_at_k:.6f}",
                    "expected_snippets": " | ".join(result.expected_snippets),
                    "forbidden_snippets": " | ".join(result.forbidden_snippets),
                    "forbidden_matches": " | ".join(result.forbidden_matches),
                }
            )


def _append_fixture_history_csv(
    output_file: Path,
    report: dict[str, Any],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_file.exists()

    with output_file.open("a", encoding="utf-8", newline="") as csv_file:
        fieldnames = [
            "generated_at",
            "fixture_file",
            "k",
            "retrieval_mode",
            "evaluated",
            "skipped",
            "hits",
            "recall_at_k",
            "mrr",
            "expected_recall_at_k",
            "precision_at_k",
            "map_at_k",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "generated_at": datetime.now(tz=UTC).isoformat(),
                "fixture_file": str(report["fixture_file"]),
                "k": int(report["k"]),
                "retrieval_mode": str(report.get("retrieval_mode", "similarity")),
                "evaluated": int(report["summary"]["evaluated"]),
                "skipped": int(report["summary"]["skipped"]),
                "hits": int(report["summary"]["hits"]),
                "recall_at_k": f"{float(report['summary']['recall_at_k']):.6f}",
                "mrr": f"{float(report['summary']['mrr']):.6f}",
                "expected_recall_at_k": f"{float(report['summary']['expected_recall_at_k']):.6f}",
                "precision_at_k": f"{float(report['summary']['precision_at_k']):.6f}",
                "map_at_k": f"{float(report['summary']['map_at_k']):.6f}",
            }
        )


def _build_runtime_eval_manager() -> object:
    config = load_runtime_config().flat
    manager = object.__new__(ConversationManager)
    manager.configs = config
    manager.rag_k = int(config.get("RAG_K", 3))
    manager.rag_k_mes = int(config.get("RAG_K_MES", manager.rag_k))
    manager.persist_directory = str(config.get("PERSIST_DIRECTORY", "./character_storage/"))
    manager.embedding_cache = str(config.get("EMBEDDING_CACHE", "./embedding_models/"))
    manager.runtime_config = SimpleNamespace(
        use_mmr=bool(config.get("USE_MMR", True)),
        rag_fetch_k=int(config.get("RAG_FETCH_K", 20)),
        lambda_mult=float(config.get("LAMBDA_MULT", 0.75)),
        rag_rerank_enabled=bool(config.get("RAG_RERANK_ENABLED", False)),
        rag_rerank_top_n=int(config.get("RAG_RERANK_TOP_N", 8)),
        rag_rerank_model=str(config.get("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
        rag_telemetry_enabled=bool(config.get("RAG_TELEMETRY_ENABLED", False)),
        rag_multi_query_enabled=bool(config.get("RAG_MULTI_QUERY_ENABLED", True)),
        rag_multi_query_max_variants=int(config.get("RAG_MULTI_QUERY_MAX_VARIANTS", 3)),
        rag_sentence_compression_enabled=bool(config.get("RAG_SENTENCE_COMPRESSION_ENABLED", True)),
        rag_sentence_compression_max_sentences=int(config.get("RAG_SENTENCE_COMPRESSION_MAX_SENTENCES", 8)),
    )
    manager._vector_client = None  # noqa: SLF001
    manager._vector_embedder = None  # noqa: SLF001
    manager._cross_encoder = None  # noqa: SLF001
    manager._vector_dbs = {}  # noqa: SLF001
    return manager


def _print_fixture_summary(metrics: dict[str, float], k: int, skipped: int) -> None:
    click.echo("\nSummary:")
    click.echo(f"  Evaluated: {int(metrics['cases'])}")
    click.echo(f"  Skipped:   {skipped}")
    click.echo(f"  Hits@{k}:  {int(metrics['hits'])}")
    click.echo(f"  Recall@{k}: {metrics['recall_at_k']:.3f}")
    click.echo(f"  MRR:       {metrics['mrr']:.3f}")
    click.echo(f"  ExpRecall@{k}: {metrics['expected_recall_at_k']:.3f}")
    click.echo(f"  Precision@{k}: {metrics['precision_at_k']:.3f}")
    click.echo(f"  MAP@{k}:      {metrics['map_at_k']:.3f}")


def _execute_fixture_evaluation(options: FixtureEvalOptions) -> FixtureEvalRun:
    fixture_file = options.fixture_file
    if not fixture_file.exists():
        msg = f"Fixture file not found: {fixture_file}"
        raise click.ClickException(msg)

    default_collection, fixture_k, dashboard_ks, cases = _load_fixture_payload(fixture_file)
    default_k = int(options.k if options.k is not None else fixture_k)
    evaluation_k = max([default_k, *dashboard_ks])

    if not cases:
        msg = "Fixture file has no cases to evaluate"
        raise click.ClickException(msg)

    app_config = load_app_config()
    script_config = load_rag_script_config(app_config)
    persist_directory = options.persist_directory or script_config.persist_directory

    embedding_device = script_config.embedding_device
    embedding_cache = script_config.embedding_cache
    embedder = HuggingFaceEmbeddings(
        model_kwargs={"device": embedding_device},
        encode_kwargs={"normalize_embeddings": False},
        cache_folder=str(Path(embedding_cache)),
    )

    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
    available_collections = {collection.name for collection in client.list_collections()}

    db_cache = {
        collection_name: Chroma(
            client=client,
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embedder,
        )
        for collection_name in available_collections
    }

    runtime_manager = _build_runtime_eval_manager() if options.retrieval_mode == "runtime" else None

    eval_context = FixtureEvalContext(
        default_collection=default_collection,
        default_k=default_k,
        evaluation_k=evaluation_k,
        dashboard_ks=dashboard_ks,
        available_collections=available_collections,
        db_cache=db_cache,
        retrieval_mode=options.retrieval_mode,
        runtime_manager=runtime_manager,
        show_failures=options.show_failures,
    )
    case_results, skipped = _run_fixture_evaluation(cases, eval_context)
    ranks = [None if result.forbidden_hit else result.rank for result in case_results if result.status == "ok"]

    metrics = _compute_fixture_metrics(ranks, default_k)
    evaluated_cases = [result for result in case_results if result.status == "ok"]
    if evaluated_cases:
        metrics["expected_recall_at_k"] = sum(result.expected_recall_at_k for result in evaluated_cases) / len(
            evaluated_cases
        )
        metrics["precision_at_k"] = sum(result.precision_at_k for result in evaluated_cases) / len(evaluated_cases)
        metrics["map_at_k"] = sum(result.average_precision_at_k for result in evaluated_cases) / len(evaluated_cases)
        metrics.update(_compute_hit_rate_by_k(ranks, dashboard_ks))
    else:
        metrics["expected_recall_at_k"] = 0.0
        metrics["precision_at_k"] = 0.0
        metrics["map_at_k"] = 0.0
        metrics.update(_compute_hit_rate_by_k([], dashboard_ks))
    report = _build_fixture_report(fixture_file, eval_context, metrics, skipped, case_results)
    return FixtureEvalRun(
        default_k=default_k,
        case_results=case_results,
        skipped=skipped,
        metrics=metrics,
        report=report,
    )


@click.group()
def cli() -> None:
    """Manage ChromaDB collections for RAG data."""


@cli.command()
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
            click.echo(f"  • {info['name']}")
            click.echo(f"    - Documents: {info['count']}")
            click.echo(f"    - Metadata: {info['metadata']}")
        else:
            click.echo(f"  • {collection.name} ({collection.count()} documents)")


@cli.command()
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
        click.echo(f"✓ Deleted collection: {collection_name}")
    except ValueError as e:
        click.secho(f"Error deleting collection: {e}", fg="red", err=True)


@cli.command()
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
        click.echo(f"  • {collection.name} ({collection.count()} documents)")

    if not yes:
        confirmation = input(f"Delete all {len(matching)} collections? (yes/no): ")
        if confirmation.lower() not in ["yes", "y"]:
            click.echo("Deletion cancelled")
            return

    for collection in matching:
        try:
            client.delete_collection(collection.name)
            click.echo(f"✓ Deleted: {collection.name}")
        except ValueError as e:
            click.secho(f"✗ Error deleting {collection.name}: {e}", fg="red", err=True)


@cli.command()
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
def test(
    collection_name: str,
    query: str,
    k: int,
    persist_directory: str | None,
    key_storage: str | None,
) -> None:
    """Test a collection with a similarity search query."""
    app_config = load_app_config()
    script_config = load_rag_script_config(app_config)
    persist_directory = persist_directory or script_config.persist_directory
    key_storage = key_storage or script_config.key_storage

    embedding_device = script_config.embedding_device
    embedding_cache = script_config.embedding_cache
    model_kwargs = {"device": embedding_device}
    encode_kwargs = {"normalize_embeddings": False}
    cache_folder = str(Path(embedding_cache))

    embedder = HuggingFaceEmbeddings(
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder=cache_folder,
    )

    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))

    info = get_collection_info(client, collection_name)
    if not info["exists"]:
        click.secho(f"Collection '{collection_name}' not found", fg="red", err=True)
        return

    click.echo(f"Testing collection: {collection_name}")
    click.echo(f"Query: {query}")
    click.echo(f"Total documents: {info['count']}")

    base_name = collection_name.replace("_mes", "")
    keyfile_path = Path(key_storage) / f"{base_name}.json"

    filters = [None]
    if keyfile_path.exists():
        click.echo(f"Loading metadata from: {keyfile_path}")
        with keyfile_path.open(encoding="utf-8") as f:
            key_data = json.load(f)
        keys = normalize_keyfile(key_data)
        matches = extract_key_matches(keys, query)
        if matches:
            click.echo(f"Matched {len(matches)} metadata key(s) from query")
            filters = build_where_filters(matches)

    db = Chroma(
        client=client,
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedder,
    )

    for filter_idx, where_filter in enumerate(filters):
        filter_label = "unfiltered" if where_filter is None else str(where_filter)
        click.echo(f"\nAttempting search #{filter_idx + 1} ({filter_label})")
        if where_filter is None:
            docs = db.similarity_search_with_score(query=query, k=k)
        else:
            docs = db.similarity_search_with_score(query=query, k=k, filter=where_filter)

        if docs:
            click.echo(f"✓ Found {len(docs)} result(s)")
            for idx, (doc, score) in enumerate(docs, 1):
                click.echo(f"\nResult {idx} (score: {score:.4f}):")
                click.echo(f"Content preview: {doc.page_content[:200]}...")
                if doc.metadata:
                    click.echo(f"Metadata keys: {list(doc.metadata.keys())[:5]}")
            return

    click.echo("No results found")


@cli.command()
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

    click.echo(f"✓ Exported {export_data['count']} documents to {output}")


@cli.command()
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


@cli.command("evaluate-fixtures")
@click.option(
    "--fixture-file",
    "fixture_file",
    default=Path("tests/fixtures/retrieval_fixtures.json"),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Fixture JSON file containing retrieval cases",
)
@click.option("--k", type=int, default=None, help="Override top-k for all fixture queries")
@click.option(
    "--retrieval-mode",
    type=click.Choice(["similarity", "runtime"], case_sensitive=False),
    default="similarity",
    show_default=True,
    help="Evaluation backend: direct similarity search or ConversationManager runtime retrieval",
)
@click.option(
    "--persist-directory",
    "-p",
    default=None,
    help="Directory where ChromaDB stores collections",
)
@click.option("--show-failures", is_flag=True, help="Print per-case failures in detail")
@click.option("--output-json", type=click.Path(path_type=Path), default=None, help="Write summary report as JSON")
@click.option("--output-csv", type=click.Path(path_type=Path), default=None, help="Write per-case report as CSV")
@click.option(
    "--history-csv",
    type=click.Path(path_type=Path),
    default=None,
    help="Append run summary metrics to a CSV history file",
)
def evaluate_fixtures(**kwargs: object) -> None:
    """Evaluate retrieval fixtures and print Recall@k and MRR summary."""
    options = FixtureEvalOptions(
        fixture_file=kwargs["fixture_file"],
        k=kwargs["k"],
        retrieval_mode=str(kwargs["retrieval_mode"]).lower(),
        persist_directory=kwargs["persist_directory"],
        show_failures=bool(kwargs["show_failures"]),
    )
    output_json = kwargs["output_json"]
    output_csv = kwargs["output_csv"]
    history_csv = kwargs["history_csv"]

    run = _execute_fixture_evaluation(options)
    _print_fixture_summary(run.metrics, run.default_k, run.skipped)
    if output_json is not None:
        _write_fixture_report_json(output_json, run.report)
        click.echo(f"Wrote JSON report: {output_json}")
    if output_csv is not None:
        _write_fixture_report_csv(output_csv, run.case_results, run.default_k)
        click.echo(f"Wrote CSV report: {output_csv}")
    if history_csv is not None:
        _append_fixture_history_csv(history_csv, run.report)
        click.echo(f"Appended history row: {history_csv}")


@cli.command("benchmark-rerank")
@click.option(
    "--fixture-file",
    "fixture_file",
    default=Path("tests/fixtures/retrieval_fixtures_rerank.json"),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Fixture JSON file focused on rerank-sensitive cases",
)
@click.option("--k", type=int, default=None, help="Override top-k for all fixture queries")
@click.option(
    "--persist-directory",
    "-p",
    default=None,
    help="Directory where ChromaDB stores collections",
)
@click.option(
    "--require-runtime-win",
    is_flag=True,
    help="Exit non-zero unless runtime Recall@k and MRR are each >= similarity",
)
def benchmark_rerank(
    fixture_file: Path,
    k: int | None,
    persist_directory: str | None,
    require_runtime_win: bool,
) -> None:
    """Run similarity vs runtime benchmark and print one-line delta summary."""
    similarity_run = _execute_fixture_evaluation(
        FixtureEvalOptions(
            fixture_file=fixture_file,
            k=k,
            retrieval_mode="similarity",
            persist_directory=persist_directory,
            show_failures=False,
        )
    )
    runtime_run = _execute_fixture_evaluation(
        FixtureEvalOptions(
            fixture_file=fixture_file,
            k=k,
            retrieval_mode="runtime",
            persist_directory=persist_directory,
            show_failures=False,
        )
    )

    recall_similarity = float(similarity_run.metrics["recall_at_k"])
    mrr_similarity = float(similarity_run.metrics["mrr"])
    recall_runtime = float(runtime_run.metrics["recall_at_k"])
    mrr_runtime = float(runtime_run.metrics["mrr"])
    recall_delta = recall_runtime - recall_similarity
    mrr_delta = mrr_runtime - mrr_similarity

    click.echo(
        "RERANK_BENCH "
        f"fixture={fixture_file} "
        f"k={runtime_run.default_k} "
        f"sim_recall={recall_similarity:.3f} "
        f"sim_mrr={mrr_similarity:.3f} "
        f"run_recall={recall_runtime:.3f} "
        f"run_mrr={mrr_runtime:.3f} "
        f"delta_recall={recall_delta:+.3f} "
        f"delta_mrr={mrr_delta:+.3f}"
    )

    if require_runtime_win and (recall_delta < 0 or mrr_delta < 0):
        msg = "Runtime benchmark regressed against similarity baseline"
        raise click.ClickException(msg)


if __name__ == "__main__":
    cli()
