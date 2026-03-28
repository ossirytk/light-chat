"""Tests for analyze_rag_text_validation and analyze_rag_text_analysis error paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.rag.analyze_rag_text_analysis import analyze_text_file
from scripts.rag.analyze_rag_text_validation import validate_metadata_file


class TestValidateMetadataFile:
    def test_file_not_found(self) -> None:
        result = validate_metadata_file(Path("/nonexistent/file.json"))
        assert result["valid"] is False
        assert result["error"] == "File not found"

    def test_invalid_json(self, tmp_path: Path) -> None:
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{invalid json", encoding="utf-8")
        result = validate_metadata_file(bad_json)
        assert result["valid"] is False
        assert "Invalid JSON" in result["error"]

    def test_data_not_list(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
        result = validate_metadata_file(f)
        assert result["valid"] is False
        assert "must be a list" in result["error"]

    def test_wrapped_content_key(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"Content": [{"uuid": "1", "text": "hello"}]}), encoding="utf-8")
        result = validate_metadata_file(f)
        assert result["valid"] is True
        assert result["total_entries"] == 1

    def test_non_dict_item(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text(json.dumps(["string_item", {"uuid": "1", "text": "ok"}]), encoding="utf-8")
        result = validate_metadata_file(f)
        assert result["valid"] is False
        assert any("Not a dictionary" in issue for issue in result["issues"])

    def test_missing_uuid(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text(json.dumps([{"text": "no uuid here"}]), encoding="utf-8")
        result = validate_metadata_file(f)
        assert result["valid"] is False
        assert any("Missing 'uuid'" in issue for issue in result["issues"])

    def test_missing_text_field(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text(json.dumps([{"uuid": "abc123"}]), encoding="utf-8")
        result = validate_metadata_file(f)
        assert result["valid"] is False
        assert any("Missing text field" in issue for issue in result["issues"])

    def test_duplicate_uuids(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text(
            json.dumps([{"uuid": "dupe", "text": "first"}, {"uuid": "dupe", "text": "second"}]),
            encoding="utf-8",
        )
        result = validate_metadata_file(f)
        assert result["valid"] is False
        assert result["duplicate_uuid_count"] == 1
        assert any("Duplicate" in issue for issue in result["issues"])

    def test_valid_file(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text(json.dumps([{"uuid": "abc", "text": "valid entry"}]), encoding="utf-8")
        result = validate_metadata_file(f)
        assert result["valid"] is True
        assert result["total_entries"] == 1
        assert result["duplicate_uuid_count"] == 0


class TestAnalyzeTextFile:
    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="File not found"):
            analyze_text_file(Path("/nonexistent/path.txt"))
