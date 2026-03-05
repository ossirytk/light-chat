"""Unit tests for retrieval fixture metric helpers."""

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.rag.manage_collections import (
    FixtureCaseResult,
    FixtureEvalContext,
    _append_fixture_history_csv,
    _average_precision_at_k,
    _build_fixture_report,
    _compute_fixture_metrics,
    _expected_match_ranks,
    _first_match_rank,
    _write_fixture_report_csv,
)


class TestRetrievalMetrics(unittest.TestCase):
    def test_first_match_rank_returns_earliest_match(self) -> None:
        results = [
            "chunk about TriOptimum",
            "chunk about SHODAN and Citadel Station",
            "chunk about style",
        ]
        rank = _first_match_rank(results, ["SHODAN", "nothing"])
        self.assertEqual(rank, 2)

    def test_first_match_rank_returns_none_without_match(self) -> None:
        rank = _first_match_rank(["a", "b"], ["x", "y"])
        self.assertIsNone(rank)

    def test_compute_fixture_metrics(self) -> None:
        metrics = _compute_fixture_metrics([1, None, 3], k=5)
        self.assertEqual(metrics["cases"], 3.0)
        self.assertEqual(metrics["hits"], 2.0)
        self.assertAlmostEqual(metrics["recall_at_k"], 2 / 3)
        self.assertAlmostEqual(metrics["mrr"], (1 + 0 + (1 / 3)) / 3)

    def test_expected_match_ranks_and_average_precision(self) -> None:
        results = [
            "SHODAN controls systems",
            "TriOptimum maintains station operations",
            "Citadel Station was compromised",
        ]
        match_ranks = _expected_match_ranks(results, ["SHODAN", "TriOptimum", "Citadel"], k=3)
        self.assertEqual(match_ranks["SHODAN"], 1)
        self.assertEqual(match_ranks["TriOptimum"], 2)
        self.assertEqual(match_ranks["Citadel"], 3)
        avg_precision = _average_precision_at_k(match_ranks, expected_total=3, k=3)
        self.assertAlmostEqual(avg_precision, (1 / 1 + 2 / 2 + 3 / 3) / 3)

    def test_average_precision_is_bounded_when_multiple_snippets_share_rank(self) -> None:
        results = [
            "SHODAN and TriOptimum both appear here",
            "Citadel appears here",
        ]
        match_ranks = _expected_match_ranks(results, ["SHODAN", "TriOptimum", "Citadel"], k=2)
        avg_precision = _average_precision_at_k(match_ranks, expected_total=3, k=2)
        self.assertLessEqual(avg_precision, 1.0)

    def test_build_fixture_report_includes_summary_and_skips(self) -> None:
        metrics = _compute_fixture_metrics([1, None], k=3)
        case_results = [
            FixtureCaseResult(
                case_id="ok_hit",
                rank=1,
                status="ok",
                query="q1",
                collection="shodan",
                expected_snippets=["SHODAN"],
                forbidden_snippets=[],
            ),
            FixtureCaseResult(
                case_id="ok_miss",
                rank=None,
                status="ok",
                query="q2",
                collection="shodan",
                expected_snippets=["TriOptimum"],
                forbidden_snippets=[],
            ),
            FixtureCaseResult(
                case_id="skipped_one",
                rank=None,
                status="missing_collection",
                query="q3",
                collection="unknown",
                expected_snippets=[],
                forbidden_snippets=[],
            ),
        ]

        report = _build_fixture_report(
            fixture_file=Path("tests/fixtures/retrieval_fixtures.json"),
            eval_context=FixtureEvalContext(
                db_cache={},
                default_collection="shodan",
                evaluation_k=3,
                dashboard_ks=[1, 3],
                available_collections={"shodan"},
                default_k=3,
                show_failures=False,
                retrieval_mode="similarity",
                runtime_manager=None,
            ),
            metrics=metrics,
            skipped=1,
            case_results=case_results,
        )
        self.assertEqual(report["summary"]["evaluated"], 2)
        self.assertEqual(report["summary"]["skipped"], 1)
        self.assertEqual(report["retrieval_mode"], "similarity")
        self.assertIn("expected_recall_at_k", report["summary"])
        self.assertIn("map_at_k", report["summary"])
        self.assertEqual(len(report["cases"]), 2)
        self.assertEqual(len(report["skipped_cases"]), 1)

    def test_write_fixture_report_csv_outputs_rows(self) -> None:
        case_results = [
            FixtureCaseResult(
                case_id="hit_case",
                rank=2,
                status="ok",
                query="Who is SHODAN?",
                collection="shodan",
                expected_snippets=["SHODAN", "AI"],
                forbidden_snippets=[],
            ),
            FixtureCaseResult(
                case_id="skip_case",
                rank=None,
                status="invalid",
                query="",
                collection="",
                expected_snippets=[],
                forbidden_snippets=[],
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "report.csv"
            _write_fixture_report_csv(output_file, case_results, default_k=5)

            with output_file.open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], "hit_case")
        self.assertEqual(rows[0]["status"], "hit")
        self.assertEqual(rows[0]["hit_at_k"], "True")
        self.assertEqual(rows[1]["status"], "skipped:invalid")

    def test_write_fixture_report_csv_marks_forbidden_match(self) -> None:
        case_results = [
            FixtureCaseResult(
                case_id="forbidden_case",
                rank=1,
                status="ok",
                query="q",
                collection="shodan",
                expected_snippets=["SHODAN"],
                forbidden_snippets=["Leonardo"],
                forbidden_matches=["Leonardo"],
                forbidden_hit=True,
            )
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "forbidden.csv"
            _write_fixture_report_csv(output_file, case_results, default_k=5)

            with output_file.open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(rows[0]["status"], "forbidden_match")
        self.assertEqual(rows[0]["forbidden_hit"], "True")

    def test_append_fixture_history_csv_writes_header_once(self) -> None:
        metrics = _compute_fixture_metrics([1, None], k=5)
        case_results = [
            FixtureCaseResult(
                case_id="ok_hit",
                rank=1,
                status="ok",
                query="q1",
                collection="shodan",
                expected_snippets=["SHODAN"],
                forbidden_snippets=[],
            ),
            FixtureCaseResult(
                case_id="ok_miss",
                rank=None,
                status="ok",
                query="q2",
                collection="shodan",
                expected_snippets=["TriOptimum"],
                forbidden_snippets=[],
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            history_file = Path(tmp_dir) / "history.csv"
            report = _build_fixture_report(
                fixture_file=Path("tests/fixtures/retrieval_fixtures.json"),
                eval_context=FixtureEvalContext(
                    db_cache={},
                    default_collection="shodan",
                    evaluation_k=5,
                    dashboard_ks=[1, 5],
                    available_collections={"shodan"},
                    default_k=5,
                    show_failures=False,
                    retrieval_mode="similarity",
                    runtime_manager=None,
                ),
                metrics=metrics,
                skipped=1,
                case_results=case_results,
            )

            _append_fixture_history_csv(history_file, report)
            _append_fixture_history_csv(history_file, report)

            with history_file.open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["fixture_file"], "tests/fixtures/retrieval_fixtures.json")
        self.assertEqual(rows[0]["k"], "5")
        self.assertEqual(rows[0]["retrieval_mode"], "similarity")
        self.assertEqual(rows[0]["evaluated"], "2")
        self.assertEqual(rows[0]["skipped"], "1")
        self.assertIn("expected_recall_at_k", rows[0])
        self.assertIn("map_at_k", rows[0])


if __name__ == "__main__":
    unittest.main()
