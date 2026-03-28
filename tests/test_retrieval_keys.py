"""Unit tests for core/retrieval_keys.py — all pure functions, no external deps."""

from __future__ import annotations

import pytest

from core.retrieval_keys import (
    _get_entry_value,
    _matches_aliases,
    build_where_filters,
    extract_key_matches,
    normalize_keyfile,
)

# ---------------------------------------------------------------------------
# normalize_keyfile
# ---------------------------------------------------------------------------


def test_normalize_keyfile_with_content_key() -> None:
    raw = {"Content": [{"uuid": "a", "text": "hello"}]}
    result = normalize_keyfile(raw)
    assert result == [{"uuid": "a", "text": "hello"}]


def test_normalize_keyfile_plain_list() -> None:
    raw = [{"uuid": "a"}, {"uuid": "b"}]
    assert normalize_keyfile(raw) == [{"uuid": "a"}, {"uuid": "b"}]


def test_normalize_keyfile_filters_non_dicts() -> None:
    raw = [{"uuid": "a"}, "not-a-dict", 42]
    assert normalize_keyfile(raw) == [{"uuid": "a"}]


def test_normalize_keyfile_non_list_returns_empty() -> None:
    assert normalize_keyfile("just a string") == []
    assert normalize_keyfile(None) == []
    assert normalize_keyfile(123) == []


def test_normalize_keyfile_empty_list() -> None:
    assert normalize_keyfile([]) == []


# ---------------------------------------------------------------------------
# _get_entry_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["text", "text_fields", "text_field", "content", "value"])
def test_get_entry_value_standard_keys(key: str) -> None:
    item = {key: "hello"}
    assert _get_entry_value(item) == "hello"


def test_get_entry_value_fallback_to_any_string() -> None:
    item = {"uuid": "x", "aliases": ["a"], "category": "c", "custom_field": "found-me"}
    assert _get_entry_value(item) == "found-me"


def test_get_entry_value_skips_reserved_keys_in_fallback() -> None:
    item = {"uuid": "only-uuid"}
    assert _get_entry_value(item) is None


def test_get_entry_value_non_string_ignored() -> None:
    item = {"text": 42, "value": ["list"], "custom": "ok"}
    assert _get_entry_value(item) == "ok"


def test_get_entry_value_empty_item() -> None:
    assert _get_entry_value({}) is None


# ---------------------------------------------------------------------------
# _matches_aliases
# ---------------------------------------------------------------------------


def test_matches_aliases_found() -> None:
    item = {"aliases": ["Aria", "Ari"]}
    assert _matches_aliases(item, "i spoke with aria yesterday") is True


def test_matches_aliases_not_found() -> None:
    item = {"aliases": ["Aria"]}
    assert _matches_aliases(item, "nothing relevant here") is False


def test_matches_aliases_non_list_returns_false() -> None:
    assert _matches_aliases({"aliases": "Aria"}, "aria") is False
    assert _matches_aliases({}, "anything") is False


def test_matches_aliases_non_string_alias_skipped() -> None:
    item = {"aliases": [123, None, "valid"]}
    assert _matches_aliases(item, "valid text") is True


# ---------------------------------------------------------------------------
# extract_key_matches
# ---------------------------------------------------------------------------


def test_extract_key_matches_empty_text() -> None:
    keys = [{"uuid": "a", "text": "hello"}]
    assert extract_key_matches(keys, "") == []


def test_extract_key_matches_value_in_text() -> None:
    keys = [{"uuid": "abc", "text": "Aria"}]
    result = extract_key_matches(keys, "talking to Aria today")
    assert result == [{"abc": "Aria"}]


def test_extract_key_matches_alias_in_text() -> None:
    keys = [{"uuid": "abc", "text": "Aria", "aliases": ["Ari"]}]
    result = extract_key_matches(keys, "talked to Ari")
    assert result == [{"abc": "Aria"}]


def test_extract_key_matches_skips_missing_uuid() -> None:
    keys = [{"text": "no uuid here"}]
    assert extract_key_matches(keys, "no uuid here") == []


def test_extract_key_matches_skips_missing_value() -> None:
    keys = [{"uuid": "abc"}]
    assert extract_key_matches(keys, "something") == []


def test_extract_key_matches_no_match() -> None:
    keys = [{"uuid": "abc", "text": "Aria"}]
    assert extract_key_matches(keys, "nothing relevant") == []


# ---------------------------------------------------------------------------
# build_where_filters
# ---------------------------------------------------------------------------


def test_build_where_filters_empty() -> None:
    assert build_where_filters([]) == [None]


def test_build_where_filters_single_match() -> None:
    match = {"abc": "Aria"}
    result = build_where_filters([match])
    assert result == [match]


def test_build_where_filters_multiple_matches() -> None:
    m1 = {"a": "Aria"}
    m2 = {"b": "Bob"}
    result = build_where_filters([m1, m2])
    assert {"$and": [m1, m2]} in result
    assert {"$or": [m1, m2]} in result
    assert len(result) == 2
