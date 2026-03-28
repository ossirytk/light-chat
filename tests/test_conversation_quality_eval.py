"""Tests for deterministic conversation quality fixture evaluation."""

import json
import tempfile
import unittest
from pathlib import Path

import click

from scripts.conversation.evaluate_quality import (
    CalibrationOptions,
    _aggregate,
    _build_calibration_report,
    _evaluate_mock,
    _load_fixture,
    _load_session_turns,
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

    def test_load_session_turns_prefers_trace_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session_dir = Path(tmp_dir)
            path = session_dir / "session_20260323T120000Z_test.json"
            path.write_text(
                json.dumps(
                    {
                        "session_name": "Shodan long run",
                        "character_name": "Shodan",
                        "quality": {
                            "persona_drift_config": {
                                "heuristic_weight": 0.6,
                            }
                        },
                        "conversation_state": {
                            "persona_drift_trace": [
                                {
                                    "turn": 1,
                                    "drift_score": 0.42,
                                    "heuristic_score": 0.70,
                                    "semantic_score": 0.30,
                                    "has_user_turn_pattern": False,
                                },
                                {
                                    "turn": 2,
                                    "drift_score": 0.39,
                                    "heuristic_score": 0.75,
                                    "semantic_score": 0.35,
                                    "has_user_turn_pattern": False,
                                },
                                {
                                    "turn": 3,
                                    "drift_score": 0.80,
                                    "heuristic_score": 0.20,
                                    "semantic_score": 0.20,
                                    "has_user_turn_pattern": True,
                                },
                            ],
                            "persona_drift_history": [0.99, 0.99, 0.99],
                        },
                    }
                ),
                encoding="utf-8",
            )

            turns, counts, current_weight = _load_session_turns(session_dir, pattern="session_*.json", min_turns=2)

        self.assertEqual(len(turns), 3)
        self.assertEqual(counts["sessions_included"], 1)
        self.assertAlmostEqual(current_weight, 0.6)
        self.assertTrue(any(turn.has_user_turn_pattern for turn in turns))
        self.assertNotEqual(turns[0].drift_score, 0.99)

    def test_build_calibration_report_recommends_thresholds_and_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            session_dir = Path(tmp_dir)
            path = session_dir / "session_20260323T120000Z_test.json"
            path.write_text(
                json.dumps(
                    {
                        "session_name": "Shodan calibration",
                        "character_name": "Shodan",
                        "quality": {
                            "persona_drift_config": {
                                "heuristic_weight": 0.6,
                            }
                        },
                        "conversation_state": {
                            "persona_drift_trace": [
                                {
                                    "turn": 1,
                                    "drift_score": 0.42,
                                    "heuristic_score": 0.70,
                                    "semantic_score": 0.30,
                                    "has_user_turn_pattern": False,
                                },
                                {
                                    "turn": 2,
                                    "drift_score": 0.39,
                                    "heuristic_score": 0.75,
                                    "semantic_score": 0.35,
                                    "has_user_turn_pattern": False,
                                },
                                {
                                    "turn": 3,
                                    "drift_score": 0.80,
                                    "heuristic_score": 0.20,
                                    "semantic_score": 0.20,
                                    "has_user_turn_pattern": True,
                                },
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            turns, counts, current_weight = _load_session_turns(session_dir, pattern="session_*.json", min_turns=2)

        report = _build_calibration_report(
            turns,
            counts=counts,
            current_heuristic_weight=current_weight,
            options=CalibrationOptions(
                warning_quantile=0.8,
                fail_quantile=0.95,
                min_threshold_gap=0.08,
                weight_candidates=[0.4, 0.6, 0.8],
            ),
        )

        recommendation = report["recommendation"]
        self.assertEqual(report["sessions_included"], 1)
        self.assertEqual(report["turns_total"], 3)
        self.assertEqual(recommendation["weight_basis"], "user_turn_pattern_separation")
        self.assertAlmostEqual(float(recommendation["heuristic_weight"]), 0.8)
        self.assertGreater(float(recommendation["fail_threshold"]), float(recommendation["warning_threshold"]))

    def _assert_conversation_fixture_schema(self, fixture_path: Path) -> None:
        """Validate conversation fixture JSON structure."""
        data = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertIsInstance(data, dict, f"Fixture must be a JSON object: {fixture_path}")
        self.assertIn("cases", data, f"Fixture must define 'cases': {fixture_path}")
        self.assertIsInstance(data["cases"], list, f"'cases' must be a list: {fixture_path}")
        self.assertGreater(len(data["cases"]), 0, f"'cases' must be non-empty: {fixture_path}")
        for case in data["cases"]:
            self.assertIsInstance(case, dict)
            self.assertIn("id", case)
            self.assertIn("turns", case)
            self.assertIsInstance(case["turns"], list)
            self.assertGreater(len(case["turns"]), 0)
            for turn in case["turns"]:
                self.assertIsInstance(turn, dict)
                self.assertIn("user", turn)
                self.assertIsInstance(turn.get("expected_contains", []), list)
                self.assertIsInstance(turn.get("forbidden_contains", []), list)

    def test_hard_fixture_schema_is_valid(self) -> None:
        self._assert_conversation_fixture_schema(Path("tests/fixtures/conversation_fixtures_hard.json"))

    def test_negative_fixture_schema_is_valid(self) -> None:
        self._assert_conversation_fixture_schema(Path("tests/fixtures/conversation_fixtures_negative.json"))

    def test_hard_fixture_loads_and_evaluates(self) -> None:
        fixture_path = Path("tests/fixtures/conversation_fixtures_hard.json")
        fixtures = _load_fixture(fixture_path)
        self.assertGreater(len(fixtures), 0)
        results = _evaluate_mock(fixtures, seed=42)
        self.assertGreater(len(results), 0)

    def test_negative_fixture_loads_and_evaluates(self) -> None:
        fixture_path = Path("tests/fixtures/conversation_fixtures_negative.json")
        fixtures = _load_fixture(fixture_path)
        self.assertGreater(len(fixtures), 0)
        results = _evaluate_mock(fixtures, seed=42)
        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()
