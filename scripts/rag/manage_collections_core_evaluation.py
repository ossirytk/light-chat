"""Fixture evaluation orchestration and report IO for collection management core."""

import csv
import json
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
from scripts.rag.manage_collections_core_collection import (
    assert_collection_fingerprint_compatible,
    build_embedding_fingerprint,
    infer_embedding_dimension,
)
from scripts.rag.manage_collections_core_metrics import (
    compute_case_match_details,
    compute_report_summaries,
    compute_run_metrics,
)
from scripts.rag.manage_collections_core_types import (
    FixtureCaseResult,
    FixtureEvalContext,
    FixtureEvalOptions,
    FixtureEvalRun,
)


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

    case_match_details = compute_case_match_details(
        chunks=chunks,
        expected_snippets=expected_snippets,
        forbidden_snippets=forbidden_snippets,
        k=eval_context.default_k,
    )
    match_ranks = case_match_details["match_ranks"]
    expected_total = int(case_match_details["expected_total"])
    matched_expected = int(case_match_details["matched_expected"])
    rank = min(match_ranks.values()) if matched_expected >= min_expected_matches and match_ranks else None
    forbidden_matches = list(case_match_details["forbidden_matches"])
    forbidden_hit = len(forbidden_matches) > 0
    expected_recall_at_k = float(case_match_details["expected_recall_at_k"])
    precision_at_k = float(case_match_details["precision_at_k"])
    average_precision_at_k = float(case_match_details["average_precision_at_k"])
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
    summary, collection_metrics = compute_report_summaries(
        evaluated_cases=evaluated_cases,
        metrics=metrics,
        skipped=skipped,
        dashboard_ks=eval_context.dashboard_ks,
        default_k=eval_context.default_k,
    )
    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "fixture_file": fixture_file.as_posix(),
        "k": eval_context.default_k,
        "dashboard_ks": eval_context.dashboard_ks,
        "retrieval_mode": eval_context.retrieval_mode,
        "summary": summary,
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
                "fixture_file": str(report["fixture_file"]).replace("\\", "/"),
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


def _build_runtime_eval_manager(embedding_model: str | None = None, embedding_device: str | None = None) -> object:
    config = load_runtime_config().flat
    resolved_embedding_model = str(config.get("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"))
    resolved_embedding_device = str(config.get("EMBEDDING_DEVICE", "cpu"))
    if embedding_model:
        resolved_embedding_model = embedding_model
    if embedding_device:
        resolved_embedding_device = embedding_device

    manager = object.__new__(ConversationManager)
    manager.configs = config
    manager.rag_k = int(config.get("RAG_K", 3))
    manager.rag_k_mes = int(config.get("RAG_K_MES", manager.rag_k))
    manager.persist_directory = str(config.get("PERSIST_DIRECTORY", "./character_storage/"))
    manager.embedding_cache = str(config.get("EMBEDDING_CACHE", "./embedding_models/"))
    manager.embedding_model = resolved_embedding_model
    manager.runtime_config = SimpleNamespace(
        embedding_device=resolved_embedding_device,
        embedding_model=resolved_embedding_model,
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

    embedding_device = options.embedding_device or script_config.embedding_device
    embedding_model = options.embedding_model or script_config.embedding_model
    embedding_cache = script_config.embedding_cache
    normalize_embeddings = True
    embedder = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": embedding_device},
        encode_kwargs={"normalize_embeddings": normalize_embeddings},
        cache_folder=str(Path(embedding_cache)),
    )

    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
    available_collections = {collection.name for collection in client.list_collections()}
    expected_fingerprint = build_embedding_fingerprint(
        embedding_model=embedding_model,
        normalize_embeddings=normalize_embeddings,
        embedding_dimension=infer_embedding_dimension(embedder),
    )
    for collection_name in available_collections:
        assert_collection_fingerprint_compatible(client, collection_name, expected_fingerprint)

    db_cache = {
        collection_name: Chroma(
            client=client,
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embedder,
        )
        for collection_name in available_collections
    }

    runtime_manager = (
        _build_runtime_eval_manager(embedding_model=embedding_model, embedding_device=embedding_device)
        if options.retrieval_mode == "runtime"
        else None
    )

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
    metrics = compute_run_metrics(case_results, dashboard_ks, default_k)
    report = _build_fixture_report(fixture_file, eval_context, metrics, skipped, case_results)
    return FixtureEvalRun(
        default_k=default_k,
        case_results=case_results,
        skipped=skipped,
        metrics=metrics,
        report=report,
    )


__all__ = [
    "_append_fixture_history_csv",
    "_build_fixture_report",
    "_build_runtime_eval_manager",
    "_evaluate_fixture_case",
    "_execute_fixture_evaluation",
    "_load_fixture_payload",
    "_print_fixture_summary",
    "_run_fixture_evaluation",
    "_write_fixture_report_csv",
    "_write_fixture_report_json",
]
