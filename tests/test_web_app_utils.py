"""Tests for web_app pure utility functions (no LLM, no server startup)."""

from __future__ import annotations

from web_app import _coerce_ui_messages, _normalize_session_name


class TestNormalizeSessionName:
    def test_none_returns_fallback(self) -> None:
        assert _normalize_session_name(None, "fallback") == "fallback"

    def test_empty_string_returns_fallback(self) -> None:
        assert _normalize_session_name("", "fallback") == "fallback"

    def test_whitespace_only_returns_fallback(self) -> None:
        assert _normalize_session_name("   ", "fallback") == "fallback"

    def test_normalizes_internal_whitespace(self) -> None:
        assert _normalize_session_name("  hello   world  ", "fb") == "hello world"

    def test_truncates_to_80_chars(self) -> None:
        long_name = "a" * 100
        result = _normalize_session_name(long_name, "fb")
        assert len(result) == 80

    def test_returns_name_unchanged_when_short(self) -> None:
        assert _normalize_session_name("My Session", "fb") == "My Session"

    def test_exactly_80_chars_not_truncated(self) -> None:
        name = "x" * 80
        assert _normalize_session_name(name, "fb") == name


class TestCoerceUiMessages:
    def test_non_list_returns_empty(self) -> None:
        assert _coerce_ui_messages("not a list") == []

    def test_none_returns_empty(self) -> None:
        assert _coerce_ui_messages(None) == []

    def test_dict_returns_empty(self) -> None:
        assert _coerce_ui_messages({"role": "user", "content": "hi"}) == []

    def test_skips_non_dict_entries(self) -> None:
        assert _coerce_ui_messages([1, "str", None]) == []

    def test_skips_invalid_roles(self) -> None:
        assert _coerce_ui_messages([{"role": "system", "content": "test"}]) == []

    def test_skips_non_string_content(self) -> None:
        assert _coerce_ui_messages([{"role": "user", "content": 123}]) == []

    def test_valid_user_message_passes_through(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        assert _coerce_ui_messages(msgs) == msgs

    def test_valid_assistant_message_passes_through(self) -> None:
        msgs = [{"role": "assistant", "content": "hi there"}]
        assert _coerce_ui_messages(msgs) == msgs

    def test_mixed_valid_and_invalid(self) -> None:
        raw: list[object] = [
            {"role": "user", "content": "good"},
            {"role": "system", "content": "dropped"},
            {"role": "assistant", "content": "also good"},
            42,
        ]
        result = _coerce_ui_messages(raw)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_empty_list_returns_empty(self) -> None:
        assert _coerce_ui_messages([]) == []
