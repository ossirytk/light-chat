"""Saveable preset profiles for runtime retrieval settings."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from core.config import ConversationRuntimeConfig

PROFILE_FIELDS: list[str] = [
    "use_mmr",
    "rag_rerank_enabled",
    "rag_sentence_compression_enabled",
    "rag_multi_query_enabled",
    "rag_k",
    "rag_k_mes",
    "debug_context",
]


class ProfileStore:
    """Persist and apply named retrieval-setting presets stored in a JSON file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def _load(self) -> dict[str, dict[str, object]]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save(self, data: dict[str, dict[str, object]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_profiles(self) -> list[str]:
        """Return sorted list of saved profile names."""
        return sorted(self._load().keys())

    def save_profile(self, name: str, config: ConversationRuntimeConfig) -> None:
        """Snapshot the profile-eligible fields from *config* under *name*."""
        data = self._load()
        data[name] = {field: getattr(config, field) for field in PROFILE_FIELDS}
        self._save(data)

    def get_profile(self, name: str) -> dict[str, object]:
        """Return the stored settings dict for *name*."""
        data = self._load()
        if name not in data:
            msg = f"Profile {name!r} not found"
            raise KeyError(msg)
        return dict(data[name])

    def apply_profile(self, name: str, config: ConversationRuntimeConfig) -> list[str]:
        """Write profile values onto *config* in place; return list of changed field names."""
        profile = self.get_profile(name)
        changed: list[str] = []
        for field, value in profile.items():
            if field not in PROFILE_FIELDS:
                continue
            current = getattr(config, field, None)
            if current != value:
                setattr(config, field, value)
                changed.append(field)
        return changed

    def delete_profile(self, name: str) -> None:
        """Remove *name* from the store (no-op if not found)."""
        data = self._load()
        data.pop(name, None)
        self._save(data)

    def current_values(self, config: ConversationRuntimeConfig) -> dict[str, object]:
        """Return current values of the profile-eligible fields from *config*."""
        return {field: getattr(config, field) for field in PROFILE_FIELDS}
