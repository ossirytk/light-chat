"""Metric and reporting math helpers for fixture evaluation."""

from scripts.rag.manage_collections_core_types import FixtureCaseResult


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


def _is_hit_at_k(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def _effective_rank(result: FixtureCaseResult) -> int | None:
    if result.forbidden_hit:
        return None
    return result.rank


def _effective_ranks(case_results: list[FixtureCaseResult]) -> list[int | None]:
    return [_effective_rank(result) for result in case_results]


def _hit_count_at_k(ranks: list[int | None], k: int) -> int:
    return sum(1 for rank in ranks if _is_hit_at_k(rank, k))


def _mean_reciprocal_rank(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    mrr_total = sum((1 / rank) if rank is not None and rank > 0 else 0.0 for rank in ranks)
    return mrr_total / len(ranks)


def _compute_fixture_metrics(ranks: list[int | None], k: int) -> dict[str, float]:
    if not ranks:
        return {
            "cases": 0.0,
            "hits": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
        }

    case_count = len(ranks)
    hits = _hit_count_at_k(ranks, k)

    return {
        "cases": float(case_count),
        "hits": float(hits),
        "recall_at_k": hits / case_count,
        "mrr": _mean_reciprocal_rank(ranks),
    }


def _compute_collection_metrics(case_results: list[FixtureCaseResult], k: int) -> dict[str, dict[str, float]]:
    by_collection: dict[str, list[FixtureCaseResult]] = {}
    for result in case_results:
        if result.status != "ok":
            continue
        by_collection.setdefault(result.collection, []).append(result)

    summary: dict[str, dict[str, float]] = {}
    for collection_name, collection_results in by_collection.items():
        ranks = _effective_ranks(collection_results)
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
    return {f"hit_rate_at_{k}": _hit_count_at_k(ranks, k) / total for k in dashboard_ks}


def _compute_macro_metrics(case_results: list[FixtureCaseResult]) -> dict[str, float]:
    if not case_results:
        return {
            "expected_recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "map_at_k": 0.0,
        }

    total = len(case_results)
    return {
        "expected_recall_at_k": sum(result.expected_recall_at_k for result in case_results) / total,
        "precision_at_k": sum(result.precision_at_k for result in case_results) / total,
        "map_at_k": sum(result.average_precision_at_k for result in case_results) / total,
    }


def _build_summary_metrics(
    case_results: list[FixtureCaseResult],
    dashboard_ks: list[int],
    default_k: int,
) -> dict[str, float]:
    evaluated_cases = [result for result in case_results if result.status == "ok"]
    ranks = _effective_ranks(evaluated_cases)
    metrics = _compute_fixture_metrics(ranks, default_k)
    metrics.update(_compute_macro_metrics(evaluated_cases))
    metrics.update(_compute_hit_rate_by_k(ranks, dashboard_ks))
    return metrics


def _report_summary_payload(metrics: dict[str, float], skipped: int) -> dict[str, float | int]:
    return {
        "evaluated": int(metrics["cases"]),
        "skipped": skipped,
        "hits": int(metrics["hits"]),
        "recall_at_k": metrics["recall_at_k"],
        "mrr": metrics["mrr"],
        "expected_recall_at_k": float(metrics.get("expected_recall_at_k", 0.0)),
        "precision_at_k": float(metrics.get("precision_at_k", 0.0)),
        "map_at_k": float(metrics.get("map_at_k", 0.0)),
    }


def compute_case_match_details(
    chunks: list[str],
    expected_snippets: list[str],
    forbidden_snippets: list[str],
    k: int,
) -> dict[str, object]:
    """Compute per-case match details used by fixture evaluation."""
    match_ranks = _expected_match_ranks(chunks, expected_snippets, k)
    expected_total = len(_clean_expected_snippets(expected_snippets))
    matched_expected = len(match_ranks)
    forbidden_matches = _find_forbidden_matches(chunks, forbidden_snippets)
    expected_recall_at_k = (matched_expected / expected_total) if expected_total > 0 else 0.0
    matched_ranks_count = len({rank for rank in match_ranks.values() if rank <= k})
    precision_at_k = (matched_ranks_count / k) if k > 0 else 0.0
    average_precision_at_k = _average_precision_at_k(match_ranks, expected_total, k)
    return {
        "match_ranks": match_ranks,
        "expected_total": expected_total,
        "matched_expected": matched_expected,
        "forbidden_matches": forbidden_matches,
        "expected_recall_at_k": expected_recall_at_k,
        "precision_at_k": precision_at_k,
        "average_precision_at_k": average_precision_at_k,
    }


def compute_report_summaries(
    evaluated_cases: list[FixtureCaseResult],
    metrics: dict[str, float],
    skipped: int,
    dashboard_ks: list[int],
    default_k: int,
) -> tuple[dict[str, float | int], dict[str, dict[str, float]]]:
    """Build top-level summary and collection summaries for fixture reports."""
    hit_rate_by_k = _compute_hit_rate_by_k(_effective_ranks(evaluated_cases), dashboard_ks)
    summary = _report_summary_payload(metrics, skipped)
    summary.update(hit_rate_by_k)
    collection_metrics = _compute_collection_metrics(evaluated_cases, default_k)
    return summary, collection_metrics


def compute_run_metrics(
    case_results: list[FixtureCaseResult],
    dashboard_ks: list[int],
    default_k: int,
) -> dict[str, float]:
    """Compute aggregate run metrics for fixture evaluation."""
    return _build_summary_metrics(case_results, dashboard_ks, default_k)


__all__ = [
    "_average_precision_at_k",
    "_compute_fixture_metrics",
    "_expected_match_ranks",
    "_first_match_rank",
    "compute_case_match_details",
    "compute_report_summaries",
    "compute_run_metrics",
]
