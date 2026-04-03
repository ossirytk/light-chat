"""Unit tests for core/preset_profiles.py."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from core.config import ConversationRuntimeConfig
from core.preset_profiles import PROFILE_FIELDS, ProfileStore

if TYPE_CHECKING:
    from pathlib import Path


def _make_config(**overrides: Any) -> ConversationRuntimeConfig:
    """Return a minimal ConversationRuntimeConfig with sensible defaults."""
    defaults: dict[str, Any] = {
        "persist_directory": "db",
        "key_storage": "keys",
        "embedding_cache": "cache",
        "embedding_device": "cpu",
        "embedding_model": "model",
        "rag_collection": "col",
        "rag_k": 5,
        "rag_k_mes": 3,
        "max_history_turns": 10,
        "use_dynamic_context": False,
        "reserved_for_response": 200,
        "min_history_turns": 2,
        "history_summarization_enabled": False,
        "history_summarization_threshold": 20,
        "history_summarization_keep_recent": 5,
        "history_summarization_max_entries": 10,
        "history_summarization_max_chars": 500,
        "check_model_context": False,
        "auto_adjust_model_context": False,
        "model_type": "llama",
        "target_vram_usage": 0.9,
        "layers": -1,
        "kv_cache_quant": "q8_0",
        "max_vector_context_chars": 2200,
        "small_talk_max_words": 6,
        "followup_rag_max_words": 10,
        "persona_drift_enabled": False,
        "persona_drift_warning_threshold": 0.4,
        "persona_drift_fail_threshold": 0.6,
        "persona_drift_history_window": 5,
        "persona_drift_heuristic_weight": 0.5,
        "persona_drift_semantic_weight": 0.5,
        "use_mmr": False,
        "rag_fetch_k": 20,
        "lambda_mult": 0.5,
        "rag_rerank_enabled": False,
        "rag_rerank_model": "",
        "rag_rerank_top_n": 3,
        "rag_telemetry_enabled": False,
        "rag_multi_query_enabled": False,
        "rag_multi_query_max_variants": 3,
        "rag_sentence_compression_enabled": False,
        "rag_sentence_compression_max_sentences": 3,
        "chunk_size_estimate": 512,
        "max_initial_retrieval": 20,
        "debug_context": False,
        "debug_prompt": False,
        "debug_prompt_fingerprint": False,
        "max_stream_chars": 4000,
        "max_silent_stream_chars": 200,
        "empty_stream_fallback": "",
        "quality_fallback_response": "",
    }
    defaults.update(overrides)
    return ConversationRuntimeConfig(**defaults)


class TestProfileStoreLoadSave:
    def test_empty_when_file_missing(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path / "profiles.json")
        assert store.list_profiles() == []

    def test_returns_empty_for_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "profiles.json"
        p.write_text("not json", encoding="utf-8")
        store = ProfileStore(p)
        assert store.list_profiles() == []

    def test_save_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "profiles.json"
        store = ProfileStore(path)
        config = _make_config()
        store.save_profile("myprofile", config)
        assert path.exists()

    def test_save_and_list(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path / "profiles.json")
        config = _make_config()
        store.save_profile("alpha", config)
        store.save_profile("beta", config)
        assert store.list_profiles() == ["alpha", "beta"]

    def test_save_snapshots_profile_fields(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path / "profiles.json")
        config = _make_config(rag_k=7, use_mmr=True)
        store.save_profile("p1", config)
        data = json.loads((tmp_path / "profiles.json").read_text(encoding="utf-8"))
        assert data["p1"]["rag_k"] == 7
        assert data["p1"]["use_mmr"] is True

    def test_save_only_stores_profile_fields(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path / "profiles.json")
        config = _make_config()
        store.save_profile("p1", config)
        data = json.loads((tmp_path / "profiles.json").read_text(encoding="utf-8"))
        for key in data["p1"]:
            assert key in PROFILE_FIELDS

    def test_overwrite_existing_profile(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path / "profiles.json")
        store.save_profile("p1", _make_config(rag_k=3))
        store.save_profile("p1", _make_config(rag_k=9))
        profile = store.get_profile("p1")
        assert profile["rag_k"] == 9


class TestProfileStoreGetDelete:
    def test_get_returns_stored_values(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path / "profiles.json")
        config = _make_config(rag_k=11, debug_context=True)
        store.save_profile("test", config)
        profile = store.get_profile("test")
        assert profile["rag_k"] == 11
        assert profile["debug_context"] is True

    def test_get_unknown_profile_raises_key_error(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path / "profiles.json")
        with pytest.raises(KeyError, match="not found"):
            store.get_profile("nonexistent")

    def test_delete_removes_profile(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path / "profiles.json")
        config = _make_config()
        store.save_profile("gone", config)
        assert "gone" in store.list_profiles()
        store.delete_profile("gone")
        assert "gone" not in store.list_profiles()

    def test_delete_nonexistent_is_noop(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path / "profiles.json")
        store.delete_profile("doesnotexist")  # should not raise


class TestProfileStoreApply:
    def test_apply_returns_new_config_instance(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path / "profiles.json")
        config = _make_config(rag_k=5)
        store.save_profile("p", config)
        new_config, _ = store.apply_profile("p", config)
        # apply_profile must return a new (or equal) instance - never mutate in place
        assert isinstance(new_config, ConversationRuntimeConfig)

    def test_apply_updates_fields(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path / "profiles.json")
        store.save_profile("fast", _make_config(rag_k=2, use_mmr=True))
        original = _make_config(rag_k=5, use_mmr=False)
        new_config, changed = store.apply_profile("fast", original)
        assert new_config.rag_k == 2
        assert new_config.use_mmr is True
        assert set(changed) == {"rag_k", "use_mmr"}

    def test_apply_does_not_mutate_original_config(self, tmp_path: Path) -> None:
        """ConversationRuntimeConfig is frozen; apply must not raise FrozenInstanceError."""
        store = ProfileStore(tmp_path / "profiles.json")
        store.save_profile("mmr_on", _make_config(use_mmr=True))
        original = _make_config(use_mmr=False)
        new_config, _ = store.apply_profile("mmr_on", original)
        # Original must be unchanged (frozen dataclass semantics)
        assert original.use_mmr is False
        assert new_config.use_mmr is True

    def test_apply_no_change_returns_same_values(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path / "profiles.json")
        config = _make_config(rag_k=5)
        store.save_profile("same", config)
        new_config, changed = store.apply_profile("same", config)
        assert changed == []
        assert new_config.rag_k == 5

    def test_apply_unknown_profile_raises_key_error(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path / "profiles.json")
        config = _make_config()
        with pytest.raises(KeyError):
            store.apply_profile("ghost", config)

    def test_apply_skips_unknown_fields_in_stored_data(self, tmp_path: Path) -> None:
        """Stored profiles with extra/unknown fields must be silently ignored."""
        path = tmp_path / "profiles.json"
        path.write_text(
            json.dumps({"p": {"rag_k": 4, "unknown_field": "ignored"}}),
            encoding="utf-8",
        )
        store = ProfileStore(path)
        original = _make_config(rag_k=7)
        new_config, changed = store.apply_profile("p", original)
        assert new_config.rag_k == 4
        assert "unknown_field" not in changed

    def test_apply_returns_dataclass_replace_result(self, tmp_path: Path) -> None:
        """Verify that apply_profile uses dataclasses.replace (frozen-safe)."""
        store = ProfileStore(tmp_path / "profiles.json")
        store.save_profile("r", _make_config(rag_k=99))
        original = _make_config(rag_k=1)
        new_config, _ = store.apply_profile("r", original)
        # dataclasses.replace produces a new instance
        assert new_config is not original
        assert new_config.rag_k == 99
        # All other fields are preserved
        assert new_config.embedding_model == original.embedding_model


class TestProfileStoreCurrentValues:
    def test_current_values_returns_all_profile_fields(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path / "profiles.json")
        config = _make_config()
        values = store.current_values(config)
        assert set(values.keys()) == set(PROFILE_FIELDS)

    def test_current_values_matches_config(self, tmp_path: Path) -> None:
        store = ProfileStore(tmp_path / "profiles.json")
        config = _make_config(rag_k=42, use_mmr=True, debug_context=True)
        values = store.current_values(config)
        assert values["rag_k"] == 42
        assert values["use_mmr"] is True
        assert values["debug_context"] is True
