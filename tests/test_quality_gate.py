"""Tests for quality_gate pure/lightweight functions."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from scripts.quality_gate import _load_baseline_summary, _print_summary_table, _step_label


class TestStepLabel:
    def test_pass_label(self) -> None:
        assert _step_label("pass") == "PASS"

    def test_fail_label(self) -> None:
        assert _step_label("fail") == "FAIL"

    def test_warn_label(self) -> None:
        assert _step_label("warn") == "WARN"

    def test_skip_label(self) -> None:
        assert _step_label("skip") == "SKIP"

    def test_unknown_status_uppercases(self) -> None:
        assert _step_label("custom") == "CUSTOM"

    def test_empty_string_uppercases(self) -> None:
        assert _step_label("") == ""


class TestLoadBaselineSummary:
    def test_returns_none_when_file_absent(self, tmp_path: Path) -> None:
        result = _load_baseline_summary(Path("fixture.json"), tmp_path)
        assert result is None

    def test_returns_none_on_invalid_json(self, tmp_path: Path) -> None:
        baseline_path = tmp_path / "fixture_baseline.json"
        baseline_path.write_text("not valid json", encoding="utf-8")
        result = _load_baseline_summary(Path("fixture.json"), tmp_path)
        assert result is None

    def test_uses_stem_for_baseline_filename(self, tmp_path: Path) -> None:
        fixture = Path("tests/fixtures/my_fixture.json")
        result = _load_baseline_summary(fixture, tmp_path)
        assert result is None

    def test_returns_none_on_exception_from_load_baseline(self, tmp_path: Path) -> None:
        # A summary value that is not a dict causes _load_baseline to raise ClickException.
        baseline_path = tmp_path / "fixture_baseline.json"
        baseline_path.write_text(json.dumps({"summary": "not-a-dict"}), encoding="utf-8")
        result = _load_baseline_summary(Path("fixture.json"), tmp_path)
        assert result is None


class TestPrintSummaryTable:
    def test_prints_headers(self) -> None:
        output: list[str] = []
        with (
            patch("click.echo", side_effect=lambda msg, **_kw: output.append(str(msg))),
            patch("click.secho", side_effect=lambda msg, **_kw: output.append(str(msg))),
        ):
            _print_summary_table([("Step A", "pass", "all good")])
        full = "\n".join(output)
        assert "Step" in full
        assert "Status" in full

    def test_renders_all_statuses(self) -> None:
        rows = [
            ("A", "pass", "ok"),
            ("B", "fail", "broken"),
            ("C", "warn", "iffy"),
            ("D", "skip", "skipped"),
        ]
        output: list[str] = []
        with (
            patch("click.echo", side_effect=lambda msg, **_kw: output.append(str(msg))),
            patch("click.secho", side_effect=lambda msg, **_kw: output.append(str(msg))),
        ):
            _print_summary_table(rows)
        full = "\n".join(output)
        for label in ("PASS", "FAIL", "WARN", "SKIP"):
            assert label in full

    def test_empty_results_prints_table(self) -> None:
        output: list[str] = []
        with (
            patch("click.echo", side_effect=lambda msg, **_kw: output.append(str(msg))),
            patch("click.secho", side_effect=lambda msg, **_kw: output.append(str(msg))),
        ):
            _print_summary_table([])
        assert len(output) >= 3
