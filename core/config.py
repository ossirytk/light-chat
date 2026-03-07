import json
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

type ConfigMap = dict[str, object]
type ConfigPath = tuple[str, ...]

CONFIG_DIR = Path("./configs/")
V2_CONFIG_FILE = "config.v2.json"

V2_TO_LEGACY_KEY_MAP: dict[ConfigPath, str] = {
    ("character", "card"): "CHARACTER_CARD",
    ("prompt", "template_file"): "PROMPT_TEMPLATE",
    ("paths", "persist_directory"): "PERSIST_DIRECTORY",
    ("paths", "key_storage"): "KEY_STORAGE",
    ("paths", "embedding_cache"): "EMBEDDING_CACHE",
    ("paths", "documents_directory"): "DOCUMENTS_DIRECTORY",
    ("embedding", "device"): "EMBEDDING_DEVICE",
    ("rag", "chunk_size"): "CHUNK_SIZE",
    ("rag", "chunk_overlap"): "CHUNK_OVERLAP",
    ("runtime", "threads"): "THREADS",
    ("rag", "collection"): "RAG_COLLECTION",
    ("rag", "k"): "RAG_K",
    ("rag", "k_mes"): "RAG_K_MES",
    ("rag", "fetch_k"): "RAG_FETCH_K",
    ("rag", "score_threshold"): "RAG_SCORE_THRESHOLD",
    ("rag", "use_mmr"): "USE_MMR",
    ("rag", "lambda_mult"): "LAMBDA_MULT",
    ("rag", "rerank", "enabled"): "RAG_RERANK_ENABLED",
    ("rag", "rerank", "model"): "RAG_RERANK_MODEL",
    ("rag", "rerank", "top_n"): "RAG_RERANK_TOP_N",
    ("rag", "telemetry", "enabled"): "RAG_TELEMETRY_ENABLED",
    ("rag", "multi_query", "enabled"): "RAG_MULTI_QUERY_ENABLED",
    ("rag", "multi_query", "max_variants"): "RAG_MULTI_QUERY_MAX_VARIANTS",
    ("debug", "prompt"): "DEBUG_PROMPT",
    ("debug", "prompt_fingerprint"): "DEBUG_PROMPT_FINGERPRINT",
    ("debug", "context"): "DEBUG_CONTEXT",
    ("context", "dynamic", "enabled"): "USE_DYNAMIC_CONTEXT",
    ("context", "budget", "reserved_for_response"): "RESERVED_FOR_RESPONSE",
    ("context", "history", "min_turns"): "MIN_HISTORY_TURNS",
    ("context", "history", "max_turns"): "MAX_HISTORY_TURNS",
    ("context", "history", "summarization", "enabled"): "HISTORY_SUMMARIZATION_ENABLED",
    ("context", "history", "summarization", "threshold_turns"): "HISTORY_SUMMARIZATION_THRESHOLD",
    ("context", "history", "summarization", "keep_recent_turns"): "HISTORY_SUMMARIZATION_KEEP_RECENT",
    ("context", "history", "summarization", "max_entries"): "HISTORY_SUMMARIZATION_MAX_ENTRIES",
    ("context", "history", "summarization", "max_chars_per_turn"): "HISTORY_SUMMARIZATION_MAX_CHARS",
    ("context", "retrieval", "chunk_size_estimate"): "CHUNK_SIZE_ESTIMATE",
    ("context", "retrieval", "max_initial_retrieval"): "MAX_INITIAL_RETRIEVAL",
    ("context", "retrieval", "max_vector_context_chars"): "MAX_VECTOR_CONTEXT_CHARS",
    ("context", "retrieval", "sentence_compression", "enabled"): "RAG_SENTENCE_COMPRESSION_ENABLED",
    ("context", "retrieval", "sentence_compression", "max_sentences"): "RAG_SENTENCE_COMPRESSION_MAX_SENTENCES",
    ("heuristics", "small_talk_max_words"): "SMALL_TALK_MAX_WORDS",
    ("heuristics", "followup_rag_max_words"): "FOLLOWUP_RAG_MAX_WORDS",
    ("generation", "max_stream_chars"): "MAX_STREAM_CHARS",
    ("generation", "max_silent_stream_chars"): "MAX_SILENT_STREAM_CHARS",
    ("generation", "hard_max_tokens"): "HARD_MAX_TOKENS",
    ("fallback", "empty_stream"): "EMPTY_STREAM_FALLBACK",
    ("fallback", "quality"): "QUALITY_FALLBACK_RESPONSE",
    ("model", "context", "check"): "CHECK_MODEL_CONTEXT",
    ("model", "context", "auto_adjust"): "AUTO_ADJUST_MODEL_CONTEXT",
    ("logging", "level"): "LOG_LEVEL",
    ("logging", "show_logs"): "SHOW_LOGS",
    ("logging", "to_file"): "LOG_TO_FILE",
    ("logging", "file"): "LOG_FILE",
    ("model", "path"): "MODEL",
    ("model", "type"): "MODEL_TYPE",
    ("model", "layers"): "LAYERS",
    ("model", "target_vram_usage"): "TARGET_VRAM_USAGE",
    ("model", "kv_cache_quant"): "KV_CACHE_QUANT",
    ("model", "n_ctx"): "N_CTX",
    ("model", "n_batch"): "N_BATCH",
    ("generation", "temperature"): "TEMPERATURE",
    ("generation", "top_p"): "TOP_P",
    ("generation", "repeat_penalty"): "REPEAT_PENALTY",
    ("generation", "top_k"): "TOP_K",
    ("generation", "last_n_tokens_size"): "LAST_N_TOKENS_SIZE",
    ("generation", "max_tokens"): "MAX_TOKENS",
}


@dataclass(frozen=True)
class RuntimeConfig:
    v2: ConfigMap
    flat: ConfigMap


@dataclass(frozen=True)
class RagScriptConfig:
    documents_directory: str
    persist_directory: str
    key_storage: str
    embedding_cache: str
    embedding_device: str
    threads: int
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class ConversationRuntimeConfig:
    persist_directory: str
    key_storage: str
    embedding_cache: str
    rag_collection: str
    rag_k: int
    rag_k_mes: int
    max_history_turns: int
    use_dynamic_context: bool
    reserved_for_response: int
    min_history_turns: int
    history_summarization_enabled: bool
    history_summarization_threshold: int
    history_summarization_keep_recent: int
    history_summarization_max_entries: int
    history_summarization_max_chars: int
    check_model_context: bool
    auto_adjust_model_context: bool
    model_type: str
    target_vram_usage: float
    layers: str | int
    kv_cache_quant: str
    max_vector_context_chars: int
    small_talk_max_words: int
    followup_rag_max_words: int
    use_mmr: bool
    rag_fetch_k: int
    lambda_mult: float
    rag_rerank_enabled: bool
    rag_rerank_model: str
    rag_rerank_top_n: int
    rag_telemetry_enabled: bool
    rag_multi_query_enabled: bool
    rag_multi_query_max_variants: int
    rag_sentence_compression_enabled: bool
    rag_sentence_compression_max_sentences: int
    chunk_size_estimate: int
    max_initial_retrieval: int
    debug_context: bool
    debug_prompt: bool
    debug_prompt_fingerprint: bool
    max_stream_chars: int
    max_silent_stream_chars: int
    empty_stream_fallback: str
    quality_fallback_response: str


def _read_json_file(path: Path) -> ConfigMap:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def _nested_lookup(data: ConfigMap, path: ConfigPath) -> object | None:
    current: object = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _flatten_v2(v2_config: ConfigMap) -> ConfigMap:
    flat: ConfigMap = {}
    for path, legacy_key in V2_TO_LEGACY_KEY_MAP.items():
        value = _nested_lookup(v2_config, path)
        if value is not None:
            flat[legacy_key] = value
    return flat


def load_runtime_config(config_dir: Path = CONFIG_DIR) -> RuntimeConfig:
    config_path = config_dir / V2_CONFIG_FILE
    if not config_path.exists():
        msg = f"Missing required config file: {config_path}"
        raise FileNotFoundError(msg)

    v2 = _read_json_file(config_path)
    v2_flat = _flatten_v2(v2)

    return RuntimeConfig(v2=v2, flat=v2_flat)


def load_app_config(config_dir: Path = CONFIG_DIR) -> ConfigMap:
    return load_runtime_config(config_dir).flat


def _get_str_value(value: object, default: str) -> str:
    if isinstance(value, str) and value:
        return value
    return default


def _get_int_value(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_float_value(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_bool_value(value: object, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def load_rag_script_config(app_config: Mapping[str, object]) -> RagScriptConfig:
    return RagScriptConfig(
        documents_directory=_get_str_value(app_config.get("DOCUMENTS_DIRECTORY"), "./rag_data/"),
        persist_directory=_get_str_value(app_config.get("PERSIST_DIRECTORY"), "./character_storage/"),
        key_storage=_get_str_value(app_config.get("KEY_STORAGE"), "./rag_data/"),
        embedding_cache=_get_str_value(app_config.get("EMBEDDING_CACHE"), "./embedding_models/"),
        embedding_device=_get_str_value(app_config.get("EMBEDDING_DEVICE"), "cpu"),
        threads=_get_int_value(app_config.get("THREADS"), 6),
        chunk_size=_get_int_value(app_config.get("CHUNK_SIZE"), 2048),
        chunk_overlap=_get_int_value(app_config.get("CHUNK_OVERLAP"), 1024),
    )


def load_conversation_runtime_config(app_config: Mapping[str, object]) -> ConversationRuntimeConfig:
    rag_k = _get_int_value(app_config.get("RAG_K"), 7)
    rag_k_mes = _get_int_value(app_config.get("RAG_K_MES"), rag_k)
    return ConversationRuntimeConfig(
        persist_directory=_get_str_value(app_config.get("PERSIST_DIRECTORY"), "./character_storage/"),
        key_storage=_get_str_value(app_config.get("KEY_STORAGE"), "./rag_data/"),
        embedding_cache=_get_str_value(app_config.get("EMBEDDING_CACHE"), "./embedding_models/"),
        rag_collection=_get_str_value(app_config.get("RAG_COLLECTION"), ""),
        rag_k=rag_k,
        rag_k_mes=rag_k_mes,
        max_history_turns=_get_int_value(app_config.get("MAX_HISTORY_TURNS"), 10),
        use_dynamic_context=_get_bool_value(app_config.get("USE_DYNAMIC_CONTEXT"), default=True),
        reserved_for_response=_get_int_value(app_config.get("RESERVED_FOR_RESPONSE"), 256),
        min_history_turns=_get_int_value(app_config.get("MIN_HISTORY_TURNS"), 1),
        history_summarization_enabled=_get_bool_value(app_config.get("HISTORY_SUMMARIZATION_ENABLED"), default=True),
        history_summarization_threshold=_get_int_value(app_config.get("HISTORY_SUMMARIZATION_THRESHOLD"), 8),
        history_summarization_keep_recent=_get_int_value(app_config.get("HISTORY_SUMMARIZATION_KEEP_RECENT"), 6),
        history_summarization_max_entries=_get_int_value(app_config.get("HISTORY_SUMMARIZATION_MAX_ENTRIES"), 12),
        history_summarization_max_chars=_get_int_value(app_config.get("HISTORY_SUMMARIZATION_MAX_CHARS"), 140),
        check_model_context=_get_bool_value(app_config.get("CHECK_MODEL_CONTEXT"), default=False),
        auto_adjust_model_context=_get_bool_value(app_config.get("AUTO_ADJUST_MODEL_CONTEXT"), default=False),
        model_type=_get_str_value(app_config.get("MODEL_TYPE"), ""),
        target_vram_usage=_get_float_value(app_config.get("TARGET_VRAM_USAGE"), 0.8),
        layers=app_config.get("LAYERS", "auto"),
        kv_cache_quant=_get_str_value(app_config.get("KV_CACHE_QUANT"), "f16"),
        max_vector_context_chars=_get_int_value(app_config.get("MAX_VECTOR_CONTEXT_CHARS"), 2200),
        small_talk_max_words=_get_int_value(app_config.get("SMALL_TALK_MAX_WORDS"), 8),
        followup_rag_max_words=_get_int_value(app_config.get("FOLLOWUP_RAG_MAX_WORDS"), 12),
        use_mmr=_get_bool_value(app_config.get("USE_MMR"), default=True),
        rag_fetch_k=_get_int_value(app_config.get("RAG_FETCH_K"), 20),
        lambda_mult=_get_float_value(app_config.get("LAMBDA_MULT"), 0.75),
        rag_rerank_enabled=_get_bool_value(app_config.get("RAG_RERANK_ENABLED"), default=False),
        rag_rerank_model=_get_str_value(app_config.get("RAG_RERANK_MODEL"), "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        rag_rerank_top_n=_get_int_value(app_config.get("RAG_RERANK_TOP_N"), 8),
        rag_telemetry_enabled=_get_bool_value(app_config.get("RAG_TELEMETRY_ENABLED"), default=False),
        rag_multi_query_enabled=_get_bool_value(app_config.get("RAG_MULTI_QUERY_ENABLED"), default=True),
        rag_multi_query_max_variants=_get_int_value(app_config.get("RAG_MULTI_QUERY_MAX_VARIANTS"), 3),
        rag_sentence_compression_enabled=_get_bool_value(
            app_config.get("RAG_SENTENCE_COMPRESSION_ENABLED"),
            default=True,
        ),
        rag_sentence_compression_max_sentences=_get_int_value(
            app_config.get("RAG_SENTENCE_COMPRESSION_MAX_SENTENCES"),
            8,
        ),
        chunk_size_estimate=_get_int_value(app_config.get("CHUNK_SIZE_ESTIMATE"), 150),
        max_initial_retrieval=_get_int_value(app_config.get("MAX_INITIAL_RETRIEVAL"), 20),
        debug_context=_get_bool_value(app_config.get("DEBUG_CONTEXT"), default=False),
        debug_prompt=_get_bool_value(app_config.get("DEBUG_PROMPT"), default=False),
        debug_prompt_fingerprint=_get_bool_value(app_config.get("DEBUG_PROMPT_FINGERPRINT"), default=False),
        max_stream_chars=_get_int_value(app_config.get("MAX_STREAM_CHARS"), 6000),
        max_silent_stream_chars=_get_int_value(app_config.get("MAX_SILENT_STREAM_CHARS"), 200),
        empty_stream_fallback=_get_str_value(
            app_config.get("EMPTY_STREAM_FALLBACK"),
            "I am unable to produce a visible response right now. Please try again.",
        ),
        quality_fallback_response=_get_str_value(
            app_config.get("QUALITY_FALLBACK_RESPONSE"),
            "I will not repeat myself. Ask your question with more specificity.",
        ),
    )


def configure_logging(app_config: Mapping[str, object]) -> None:
    show_logs = bool(app_config.get("SHOW_LOGS", True))
    log_level = str(app_config.get("LOG_LEVEL", "DEBUG")).upper()
    log_to_file = bool(app_config.get("LOG_TO_FILE", True))
    log_file = str(app_config.get("LOG_FILE", "./logs/light-chat.log"))

    logger.remove()

    if log_to_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_path, level=log_level, rotation="10 MB", retention=5)

    if show_logs:
        logging.basicConfig(level=log_level)
        logger.add(sys.stderr, level=log_level)
    else:
        logging.disable(logging.CRITICAL)
