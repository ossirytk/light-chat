"""Tests for RAG data quality tools: coverage scoring, message linting, category thresholds."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.rag.analyze_rag_coverage import (
    CoverageMetrics,
    extract_coverage_metrics,
    format_coverage_report,
    load_metadata_file,
)
from scripts.rag.lint_message_examples import (
    MessageExamplesLinter,
    SeverityLevel,
    format_lint_report,
    lint_file_path,
)
from scripts.rag.manage_collections_config_categories import (
    apply_threshold,
    create_config,
    get_default_config,
)

# ---------------------------------------------------------------------------
# Coverage Scoring Tests
# ---------------------------------------------------------------------------


class TestCoverageMetrics(unittest.TestCase):
    """Unit tests for extract_coverage_metrics."""

    def test_empty_metadata_returns_zero_coverage(self) -> None:
        metrics = extract_coverage_metrics("Hello world", [])
        self.assertEqual(metrics.source_coverage_ratio, 0.0)
        self.assertEqual(metrics.entities_count, 0)

    def test_full_match_returns_high_coverage(self) -> None:
        source = "Leonardo da Vinci"
        metadata = [{"uuid": "abc", "text": "Leonardo da Vinci", "category": "character"}]
        metrics = extract_coverage_metrics(source, metadata)
        self.assertGreater(metrics.source_coverage_ratio, 0.9)
        self.assertEqual(metrics.entities_count, 1)

    def test_partial_match_returns_partial_coverage(self) -> None:
        source = "Leonardo da Vinci was a great inventor and painter."
        metadata = [{"uuid": "abc", "text": "Leonardo da Vinci", "category": "character"}]
        metrics = extract_coverage_metrics(source, metadata)
        self.assertGreater(metrics.source_coverage_ratio, 0.0)
        self.assertLess(metrics.source_coverage_ratio, 1.0)

    def test_no_match_returns_zero_coverage(self) -> None:
        source = "Hello world foo bar baz"
        metadata = [{"uuid": "abc", "text": "TriOptimum", "category": "faction"}]
        metrics = extract_coverage_metrics(source, metadata)
        self.assertEqual(metrics.covered_chars, 0)
        self.assertIn("TriOptimum", [v["text"] for v in metrics.entity_coverage.values()])

    def test_category_distribution_computed(self) -> None:
        source = "Alpha Beta Gamma Delta"
        metadata = [
            {"uuid": "1", "text": "Alpha", "category": "faction"},
            {"uuid": "2", "text": "Beta", "category": "faction"},
            {"uuid": "3", "text": "Gamma", "category": "technology"},
        ]
        metrics = extract_coverage_metrics(source, metadata)
        self.assertEqual(metrics.category_distribution["faction"], 2)
        self.assertEqual(metrics.category_distribution["technology"], 1)

    def test_entity_found_in_coverage(self) -> None:
        source = "Shodan controls the station."
        metadata = [{"uuid": "x1", "text": "Shodan", "category": "character"}]
        metrics = extract_coverage_metrics(source, metadata)
        found_item = metrics.entity_coverage["x1"]
        self.assertTrue(found_item["found"])

    def test_entity_not_found_tracked(self) -> None:
        source = "Nothing relevant here."
        metadata = [{"uuid": "x2", "text": "TriOptimum", "category": "faction"}]
        metrics = extract_coverage_metrics(source, metadata)
        self.assertFalse(metrics.entity_coverage["x2"]["found"])

    def test_case_insensitive_matching(self) -> None:
        source = "SHODAN controls everything."
        metadata = [{"uuid": "x3", "text": "Shodan", "category": "character"}]
        metrics = extract_coverage_metrics(source, metadata)
        self.assertGreater(metrics.source_coverage_ratio, 0.0)


class TestFormatCoverageReport(unittest.TestCase):
    """Tests for coverage report formatting."""

    def _make_metrics(self, ratio: float) -> CoverageMetrics:
        return CoverageMetrics(
            entities_count=5,
            source_coverage_ratio=ratio,
            total_source_chars=1000,
            covered_chars=int(ratio * 1000),
            unmapped_segments=["unmapped text here"],
            category_distribution={"faction": 3, "technology": 2},
            entity_coverage={},
        )

    def test_pass_status_when_above_threshold(self) -> None:
        metrics = self._make_metrics(0.80)
        report = format_coverage_report(metrics, threshold=0.75)
        self.assertIn("PASS", report)
        self.assertNotIn("FAIL", report)

    def test_fail_status_when_below_threshold(self) -> None:
        metrics = self._make_metrics(0.60)
        report = format_coverage_report(metrics, threshold=0.75)
        self.assertIn("FAIL", report)

    def test_report_includes_category_breakdown(self) -> None:
        metrics = self._make_metrics(0.80)
        report = format_coverage_report(metrics)
        self.assertIn("faction", report)
        self.assertIn("technology", report)


class TestLoadMetadataFile(unittest.TestCase):
    """Tests for metadata file loading and normalization."""

    def test_loads_plain_list(self) -> None:
        data = [{"uuid": "abc", "text": "Test", "category": "faction"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)
        try:
            result = load_metadata_file(path)
            self.assertEqual(result, data)
        finally:
            path.unlink()

    def test_loads_wrapped_content(self) -> None:
        data = {"Content": [{"uuid": "abc", "text": "Test", "category": "faction"}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)
        try:
            result = load_metadata_file(path)
            self.assertEqual(result, data["Content"])
        finally:
            path.unlink()

    def test_raises_on_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_metadata_file(Path("/nonexistent/path.json"))

    def test_raises_on_wrong_shape(self) -> None:
        data = {"not_a_list": True}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)
        try:
            with self.assertRaises(TypeError):
                load_metadata_file(path)
        finally:
            path.unlink()


# ---------------------------------------------------------------------------
# Message Examples Linting Tests
# ---------------------------------------------------------------------------


class TestMessageExamplesLinter(unittest.TestCase):
    """Unit tests for the message examples linter."""

    VALID_HEADER = "<!-- character: Test | source: Test source | version: 1 | edited: 2024-01-01 -->"

    def _write_tmp(self, content: str) -> Path:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            return Path(f.name)

    def tearDown(self) -> None:
        # Clean up any temp files (best effort)
        pass

    def test_valid_file_passes(self) -> None:
        content = f"{self.VALID_HEADER}\n\n[USER]: Hello?\n\n[ASSISTANT]: Hi there.\n"
        path = self._write_tmp(content)
        try:
            report = lint_file_path(path)
            self.assertTrue(report.valid)
            self.assertEqual(len([v for v in report.violations if v.severity == SeverityLevel.ERROR]), 0)
        finally:
            path.unlink()

    def test_missing_header_is_error(self) -> None:
        content = "[USER]: Hello?\n\n[ASSISTANT]: Hi there.\n"
        path = self._write_tmp(content)
        try:
            report = lint_file_path(path)
            self.assertFalse(report.valid)
            rule_ids = [v.rule_id for v in report.violations]
            self.assertIn("missing_header", rule_ids)
        finally:
            path.unlink()

    def test_old_label_format_is_error(self) -> None:
        content = f"{self.VALID_HEADER}\n\nUser: Hello?\n\nAssistant: Hi there.\n"
        path = self._write_tmp(content)
        try:
            report = lint_file_path(path)
            rule_ids = [v.rule_id for v in report.violations]
            self.assertIn("old_label_format", rule_ids)
        finally:
            path.unlink()

    def test_auto_fix_old_labels(self) -> None:
        content = f"{self.VALID_HEADER}\n\nUser: Hello?\n\nAssistant: Hi there.\n"
        path = self._write_tmp(content)
        try:
            linter = MessageExamplesLinter(auto_fix=True)
            report = linter.lint_file(path)
            self.assertTrue(report.auto_fixed)
            fixed_content = path.read_text(encoding="utf-8")
            self.assertIn("[USER]:", fixed_content)
            self.assertIn("[ASSISTANT]:", fixed_content)
            self.assertNotIn("User:", fixed_content)
        finally:
            path.unlink()

    def test_malformed_section_break_is_warning(self) -> None:
        content = f"{self.VALID_HEADER}\n\n[USER]: Hello?\n\n[ASSISTANT]: Hi.\n\n--\n\n[USER]: More?\n"
        path = self._write_tmp(content)
        try:
            report = lint_file_path(path)
            violations = [v for v in report.violations if v.rule_id == "malformed_section_break"]
            self.assertTrue(len(violations) > 0)
            self.assertEqual(violations[0].severity, SeverityLevel.WARNING)
        finally:
            path.unlink()

    def test_empty_file_is_error(self) -> None:
        path = self._write_tmp("")
        try:
            report = lint_file_path(path)
            self.assertFalse(report.valid)
            self.assertEqual(report.violations[0].rule_id, "empty_file")
        finally:
            path.unlink()

    def test_format_report_shows_pass(self) -> None:
        content = f"{self.VALID_HEADER}\n\n[USER]: Hello?\n\n[ASSISTANT]: Hi.\n"
        path = self._write_tmp(content)
        try:
            report = lint_file_path(path)
            formatted = format_lint_report(report)
            self.assertIn("PASS", formatted)
        finally:
            path.unlink()

    def test_format_report_shows_fail_with_violations(self) -> None:
        content = "[USER]: Hello?\n\n[ASSISTANT]: Hi.\n"
        path = self._write_tmp(content)
        try:
            report = lint_file_path(path)
            formatted = format_lint_report(report)
            self.assertIn("FAIL", formatted)
            self.assertIn("missing_header", formatted)
        finally:
            path.unlink()


# ---------------------------------------------------------------------------
# Category Threshold Configuration Tests
# ---------------------------------------------------------------------------


class TestCategoryThresholdConfig(unittest.TestCase):
    """Tests for category confidence threshold configuration."""

    def test_default_config_values(self) -> None:
        config = get_default_config()
        self.assertEqual(config.strict_threshold, 0.75)
        self.assertFalse(config.allow_unassigned_categories)

    def test_create_config_overrides_threshold(self) -> None:
        config = create_config(strict_threshold=0.85)
        self.assertEqual(config.strict_threshold, 0.85)

    def test_create_config_overrides_unassigned(self) -> None:
        config = create_config(allow_unassigned_categories=True)
        self.assertTrue(config.allow_unassigned_categories)

    def test_create_config_uses_defaults_when_none(self) -> None:
        config = create_config()
        self.assertEqual(config.strict_threshold, 0.75)
        self.assertFalse(config.allow_unassigned_categories)

    def test_invalid_threshold_raises(self) -> None:
        with self.assertRaises(ValueError):
            create_config(strict_threshold=1.5)

    def test_invalid_negative_threshold_raises(self) -> None:
        with self.assertRaises(ValueError):
            create_config(strict_threshold=-0.1)

    def test_apply_threshold_passes_above(self) -> None:
        config = create_config(strict_threshold=0.75)
        self.assertTrue(apply_threshold(0.80, config))
        self.assertTrue(apply_threshold(0.75, config))  # Boundary: inclusive

    def test_apply_threshold_fails_below(self) -> None:
        config = create_config(strict_threshold=0.75)
        self.assertFalse(apply_threshold(0.74, config))
        self.assertFalse(apply_threshold(0.50, config))

    def test_strict_threshold_boundary(self) -> None:
        config = create_config(strict_threshold=0.90)
        self.assertTrue(apply_threshold(0.90, config))
        self.assertFalse(apply_threshold(0.89, config))


if __name__ == "__main__":
    unittest.main()
