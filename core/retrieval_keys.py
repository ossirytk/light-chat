from core.retrieval_shared_types import KeyItem, KeyMatch, WhereFilter


def normalize_keyfile(raw_keys: object) -> list[KeyItem]:
    if isinstance(raw_keys, dict) and "Content" in raw_keys:
        raw_keys = raw_keys["Content"]
    if not isinstance(raw_keys, list):
        return []
    return [item for item in raw_keys if isinstance(item, dict)]


def _get_entry_value(item: KeyItem) -> str | None:
    text_keys = ("text", "text_fields", "text_field", "content", "value")
    for key in text_keys:
        candidate = item.get(key)
        if isinstance(candidate, str):
            return candidate
    for key, candidate in item.items():
        if key in ("uuid", "aliases", "category"):
            continue
        if isinstance(candidate, str):
            return candidate
    return None


def _matches_aliases(item: KeyItem, text_lower: str) -> bool:
    aliases = item.get("aliases")
    if not isinstance(aliases, list):
        return False
    return any(isinstance(alias, str) and alias.lower() in text_lower for alias in aliases)


def extract_key_matches(keys: list[KeyItem], text: str) -> list[KeyMatch]:
    if not text:
        return []
    text_lower = text.lower()
    matches: list[KeyMatch] = []
    for item in keys:
        uuid = item.get("uuid")
        if not isinstance(uuid, str):
            continue
        value = _get_entry_value(item)
        if not isinstance(value, str):
            continue
        if value.lower() in text_lower or _matches_aliases(item, text_lower):
            matches.append({uuid: value})
    return matches


def build_where_filters(matches: list[KeyMatch]) -> list[WhereFilter]:
    if not matches:
        return [None]
    if len(matches) == 1:
        return [matches[0]]
    return [{"$and": matches}, {"$or": matches}]
