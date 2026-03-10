import json
from pathlib import Path

from core.retrieval_keys import extract_key_matches, normalize_keyfile
from core.retrieval_shared_types import KeyMatch


class ConversationRetrievalKeyfileMixin:
    def _get_key_matches(self, query: str, collection_name: str) -> list[KeyMatch]:
        if not query:
            return []
        keyfile_path = Path(self.key_storage) / f"{collection_name}.json"
        if not keyfile_path.exists():
            return []
        with keyfile_path.open(encoding="utf-8") as key_file:
            key_data = json.load(key_file)
        keys = normalize_keyfile(key_data)
        return extract_key_matches(keys, query)
