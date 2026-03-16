import contextlib
import logging
import re
from collections import deque

from loguru import logger

from core.config import load_conversation_runtime_config, load_runtime_config
from core.context_manager import ApproximateTokenCounter, ContextManager
from core.conversation_model_setup_mixin import ConversationModelSetupMixin
from core.conversation_prompt_history_mixin import ConversationPromptHistoryMixin
from core.conversation_response_mixin import ConversationResponseMixin
from core.conversation_retrieval_mixin import ConversationRetrievalMixin
from core.persona_drift import PersonaAnchor, PersonaDriftScorer


class ConversationManager(
    ConversationModelSetupMixin,
    ConversationRetrievalMixin,
    ConversationPromptHistoryMixin,
    ConversationResponseMixin,
):
    _LOW_QUALITY_CONTEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\b(main antagonist of (the )?system shock series)\b", re.IGNORECASE),
        re.compile(r"\b(sits before user|stands before user|before user\.)\b", re.IGNORECASE),
        re.compile(r"\b(piercing green eyes|shimmering wires|cold demeanor)\b", re.IGNORECASE),
    )
    _SMALL_TALK_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\b(hi|hello|hey)\b", re.IGNORECASE),
        re.compile(r"\bhow are you\b", re.IGNORECASE),
        re.compile(r"\bhow('s| is) it going\b", re.IGNORECASE),
        re.compile(r"\bwhat'?s up\b", re.IGNORECASE),
        re.compile(r"\bgood (morning|afternoon|evening|night)\b", re.IGNORECASE),
    )
    _MARKDOWN_HEADING_RE: re.Pattern[str] = re.compile(r"^#{2,3}\s+(.+?)\s*$")
    _SENTENCE_SPLIT_RE: re.Pattern[str] = re.compile(r"(?<=[.!?])\s+")
    _CHUNK_SIGNATURE_LINES: int = 8
    _DEFAULT_MAX_VECTOR_CONTEXT_CHARS: int = 2200
    _MIN_DYNAMIC_CONTENT_TOKENS: int = 500
    _MIN_QUERY_TERM_LEN: int = 2
    _COMPACT_QUERY_TERM_COUNT: int = 3
    _QUERY_STOPWORDS: frozenset[str] = frozenset(
        {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "how",
            "i",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "that",
            "the",
            "this",
            "to",
            "was",
            "what",
            "when",
            "where",
            "who",
            "why",
            "with",
            "you",
            "your",
        }
    )

    def __init__(self) -> None:
        # Character card details
        self.character_name: str = ""
        self.description: str = ""
        self.scenario: str = ""
        self.mes_example: str = ""
        self.first_message: str = ""
        self.voice_instructions: str = ""
        self._greeting_in_history: bool = False
        self.llama_input: str = ""
        self.llama_instruction: str = ""
        self.llama_response: str = ""
        self.llama_endtoken: str = ""
        self.add_bos: bool = False

        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)

        # Init things
        self.configs = self.load_configs()
        self.runtime_config = load_conversation_runtime_config(self.configs)
        self.prompt = self.parse_prompt()
        self.llm_model = self.instantiate_llm()
        self.persist_directory = self.runtime_config.persist_directory
        self.key_storage = self.runtime_config.key_storage
        self.embedding_cache = self.runtime_config.embedding_cache
        self.embedding_model = self.runtime_config.embedding_model
        self.rag_k = self.runtime_config.rag_k
        self.rag_k_mes = self.runtime_config.rag_k_mes
        self.rag_collection = self.runtime_config.rag_collection or self._sanitize_collection_name(self.character_name)
        _default_max_history = self.runtime_config.max_history_turns
        _raw_max_history = self.configs.get("MAX_HISTORY_TURNS", _default_max_history)
        try:
            max_history = int(_raw_max_history)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid MAX_HISTORY_TURNS value {!r} in configuration; falling back to default {}",
                _raw_max_history,
                _default_max_history,
            )
            max_history = _default_max_history
        # Write validated value back so ContextManager (initialised below) reads the same limit
        self.configs["MAX_HISTORY_TURNS"] = max_history
        self.user_message_history: deque[str] = deque(maxlen=max_history)
        self.ai_message_history: deque[str] = deque(maxlen=max_history)
        self.history_summaries: deque[str] = deque(maxlen=self.runtime_config.history_summarization_max_entries)
        self._last_summary_topic_terms: set[str] = set()
        drift_window = max(1, self.runtime_config.persona_drift_history_window)
        self.persona_drift_history: deque[float] = deque(maxlen=drift_window)
        self.last_persona_drift: dict[str, object] | None = None
        self.persona_drift_scorer = PersonaDriftScorer(
            PersonaAnchor(
                character_name=self.character_name,
                description=self.description,
                scenario=self.scenario,
                voice_instructions=self.voice_instructions,
            ),
            heuristic_weight=self.runtime_config.persona_drift_heuristic_weight,
            semantic_weight=self.runtime_config.persona_drift_semantic_weight,
        )
        self.last_retrieval_debug: dict[str, object] = {
            "collection": self.rag_collection,
            "key_match_count": 0,
            "main": {"mode": "unknown", "returned": 0, "candidates": 0, "queries": 0, "rerank_applied": False},
            "mes": {"mode": "unknown", "returned": 0, "candidates": 0, "queries": 0, "rerank_applied": False},
            "cleanup": {"main": 0, "mes": 0, "cross_removed": 0},
        }
        self._vector_client: object | None = None
        self._vector_embedder: object | None = None
        self._cross_encoder: object | None = None
        self._vector_dbs: dict[str, object] = {}

        # Initialize context manager for dynamic context window allocation
        self.context_manager = self._initialize_context_manager()
        self.use_dynamic_context = self.runtime_config.use_dynamic_context

    def __del__(self) -> None:
        """Cleanup the LlamaCpp model when the ConversationManager is destroyed."""
        if hasattr(self, "llm_model") and self.llm_model is not None:
            with contextlib.suppress(Exception):
                self.llm_model.__del__()

    def load_configs(self) -> dict:
        runtime_config = load_runtime_config()
        merged_config = runtime_config.flat
        logger.debug("Loaded config type: {}", type(merged_config))
        return merged_config

    def _sanitize_collection_name(self, name: str) -> str:
        sanitized = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
        return sanitized.strip("_") or "default"

    def _initialize_context_manager(self) -> ContextManager:
        """Initialize context manager with model's context window."""
        context_window = 4096  # Default fallback

        # Try to get actual context window from model
        try:
            client = getattr(self.llm_model, "client", None)
            if client is not None:
                n_ctx = self._read_llama_ctx_value(client, "n_ctx")
                if n_ctx is not None:
                    context_window = n_ctx
                    logger.debug("Detected model context window: {} tokens", context_window)
        except Exception:
            logger.debug("Could not detect context window, using default: {}", context_window)

        reserved_for_response = self.runtime_config.reserved_for_response

        return ContextManager(
            context_window=context_window,
            token_counter=ApproximateTokenCounter(),
            reserved_for_response=reserved_for_response,
            min_history_turns=self.runtime_config.min_history_turns,
            max_history_turns=int(self.configs.get("MAX_HISTORY_TURNS", self.runtime_config.max_history_turns)),
        )
