"""Tests for deterministic conversation quality fixture evaluation."""

import json
import tempfile
import unittest
from pathlib import Path

import click

from scripts.conversation.evaluate_quality import (
    _aggregate,
    _evaluate_mock,
    _load_fixture,
    _to_report,
)


class TestConversationQualityEval(unittest.TestCase):
    """Regression checks for fixture parsing and deterministic mock runs."""

    def test_load_fixture_requires_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad.json"
            path.write_text(json.dumps({"persona": "Shodan", "cases": []}), encoding="utf-8")
            with self.assertRaises(click.ClickException):
                _load_fixture(path)

    def test_mock_run_is_deterministic_for_same_seed(self) -> None:
        fixture_path = Path("tests/fixtures/conversation_fixtures.json")
        fixtures = _load_fixture(fixture_path)
        run_a = _evaluate_mock(fixtures, seed=42)
        run_b = _evaluate_mock(fixtures, seed=42)
        self.assertEqual(len(run_a), len(run_b))
        self.assertEqual([item.drift_score for item in run_a], [item.drift_score for item in run_b])
        self.assertEqual([item.expected_ratio for item in run_a], [item.expected_ratio for item in run_b])

    def test_report_contains_summary_and_turns(self) -> None:
        fixture_path = Path("tests/fixtures/conversation_fixtures.json")
        fixtures = _load_fixture(fixture_path)
        turn_results = _evaluate_mock(fixtures, seed=42)
        summary = _aggregate(turn_results)
        report = _to_report(
            fixture_file=fixture_path,
            mode="mock",
            seed=42,
            results=turn_results,
            summary=summary,
        )
        self.assertIn("summary", report)
        self.assertIn("turn_results", report)
        self.assertGreater(len(report["turn_results"]), 0)


if __name__ == "__main__":
    unittest.main()
