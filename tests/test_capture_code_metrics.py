"""Tests for capture_code_metrics pure functions."""

from __future__ import annotations

from scripts.quality.capture_code_metrics import _parse_complexity, _parse_coverage


class TestParseCoverage:
    def test_empty_data_returns_zeros(self) -> None:
        result = _parse_coverage({})
        assert result["total_pct"] == 0.0
        assert result["files"] == {}

    def test_extracts_total_pct(self) -> None:
        data = {"totals": {"percent_covered": 75.5}, "files": {}}
        result = _parse_coverage(data)
        assert result["total_pct"] == 75.5

    def test_extracts_file_summary(self) -> None:
        data = {
            "totals": {"percent_covered": 50.0},
            "files": {
                "src/foo.py": {"summary": {"percent_covered": 80.0, "missing_lines": 2}}
            },
        }
        result = _parse_coverage(data)
        assert "src/foo.py" in result["files"]
        assert result["files"]["src/foo.py"]["pct"] == 80.0
        assert result["files"]["src/foo.py"]["missing_lines"] == 2

    def test_rounds_to_two_decimal_places(self) -> None:
        data = {"totals": {"percent_covered": 33.333333}, "files": {}}
        result = _parse_coverage(data)
        assert result["total_pct"] == 33.33

    def test_threshold_is_present(self) -> None:
        result = _parse_coverage({})
        assert "threshold" in result
        assert isinstance(result["threshold"], int)

    def test_multiple_files(self) -> None:
        data = {
            "totals": {"percent_covered": 60.0},
            "files": {
                "a.py": {"summary": {"percent_covered": 40.0, "missing_lines": 10}},
                "b.py": {"summary": {"percent_covered": 80.0, "missing_lines": 2}},
            },
        }
        result = _parse_coverage(data)
        assert len(result["files"]) == 2

    def test_missing_file_summary_fields_default_to_zero(self) -> None:
        data = {"totals": {"percent_covered": 50.0}, "files": {"x.py": {"summary": {}}}}
        result = _parse_coverage(data)
        assert result["files"]["x.py"]["pct"] == 0.0
        assert result["files"]["x.py"]["missing_lines"] == 0


class TestParseComplexity:
    def test_empty_data(self) -> None:
        result = _parse_complexity({})
        assert result["avg"] == 0.0
        assert result["max"] == 0
        assert result["violations"] == []

    def test_no_violations_for_low_complexity(self) -> None:
        data = {"foo.py": [{"complexity": 2, "rank": "A", "name": "fn", "type": "function"}]}
        result = _parse_complexity(data)
        assert result["violations"] == []
        assert result["avg"] == 2.0

    def test_captures_violations_above_threshold(self) -> None:
        data = {
            "foo.py": [
                {"complexity": 15, "rank": "C", "name": "big_fn", "type": "function"},
                {"complexity": 2, "rank": "A", "name": "small_fn", "type": "function"},
            ]
        }
        result = _parse_complexity(data)
        assert len(result["violations"]) == 1
        assert result["violations"][0]["name"] == "big_fn"

    def test_max_and_avg(self) -> None:
        data = {
            "foo.py": [
                {"complexity": 4, "rank": "A", "name": "a", "type": "function"},
                {"complexity": 10, "rank": "C", "name": "b", "type": "function"},
            ]
        }
        result = _parse_complexity(data)
        assert result["max"] == 10
        assert result["avg"] == 7.0

    def test_violations_sorted_descending(self) -> None:
        data = {
            "foo.py": [
                {"complexity": 5, "rank": "C", "name": "medium", "type": "function"},
                {"complexity": 20, "rank": "D", "name": "large", "type": "function"},
                {"complexity": 8, "rank": "C", "name": "larger", "type": "function"},
            ]
        }
        result = _parse_complexity(data)
        complexities = [v["complexity"] for v in result["violations"]]
        assert complexities == sorted(complexities, reverse=True)

    def test_b_rank_is_not_a_violation(self) -> None:
        data = {"foo.py": [{"complexity": 6, "rank": "B", "name": "fn", "type": "function"}]}
        result = _parse_complexity(data)
        assert result["violations"] == []

    def test_violation_includes_required_fields(self) -> None:
        data = {"foo.py": [{"complexity": 12, "rank": "C", "name": "myfn", "type": "method"}]}
        result = _parse_complexity(data)
        v = result["violations"][0]
        assert v["file"] == "foo.py"
        assert v["name"] == "myfn"
        assert v["type"] == "method"
        assert v["complexity"] == 12
        assert v["rank"] == "C"

    def test_multiple_files_aggregated(self) -> None:
        data = {
            "a.py": [{"complexity": 3, "rank": "A", "name": "x", "type": "function"}],
            "b.py": [{"complexity": 5, "rank": "A", "name": "y", "type": "function"}],
        }
        result = _parse_complexity(data)
        assert result["avg"] == 4.0
        assert result["max"] == 5
