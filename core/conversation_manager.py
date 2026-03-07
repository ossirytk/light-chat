import asyncio
import contextlib
import hashlib
import json
import logging
import re
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate, load_prompt
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
from sentence_transformers import CrossEncoder

from core.config import load_conversation_runtime_config, load_runtime_config
from core.context_manager import ApproximateTokenCounter, ContextManager
from core.gpu_utils import get_n_gpu_layers

type KeyItem = dict[str, object]
type KeyMatch = dict[str, str]
type WhereFilter = dict[str, object] | None
type RetrievalTrace = dict[str, object]


class UnknownModelTypeError(Exception):
    """Raised when an unsupported model type is configured."""


class ModelLoadError(Exception):
    """Raised when a model fails to load."""


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


class ConversationManager:
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
        self._vector_client: chromadb.PersistentClient | None = None
        self._vector_embedder: HuggingFaceEmbeddings | None = None
        self._cross_encoder: CrossEncoder | None = None
        self._vector_dbs: dict[str, Chroma] = {}

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

    def _replace_template_variables(self, text: str, user_name: str = "User", char_name: str = "") -> str:
        """Replace common template variables in character card text."""
        text = text.replace("{{user}}", user_name)
        text = text.replace("{{User}}", user_name)
        text = text.replace("{{char}}", char_name)
        return text.replace("{{Char}}", char_name)

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

    def _get_model_format_tokens(self, model_type: str) -> dict[str, str]:
        """Get format tokens for the specified model type."""
        formats = {
            "alpaca": {
                "instruction": "### Instruction:",
                "input": "### Input:",
                "response": "### Response:",
                "endtoken": "",
            },
            "mistral": {
                "instruction": "<s>[INST]",
                "input": "",
                "response": "[/INST]",
                "endtoken": "",
            },
            "llama": {
                "instruction": "<s>[INST]",
                "input": "",
                "response": "[/INST]",
                "endtoken": "</s>",
            },
            "llama2": {
                "instruction": "<s>[INST]",
                "input": "",
                "response": "[/INST]",
                "endtoken": "</s>",
            },
            "chatml": {
                "instruction": "<|system|>",
                "input": "<|user|>",
                "response": "<|assistant|>",
                "endtoken": "</s>",
            },
            "qwen": {
                "instruction": "<|im_start|>system\n",
                "input": "<|im_start|>user\n",
                "response": "<|im_start|>assistant\n",
                "endtoken": "<|im_end|>\n",
            },
            "qwen2": {
                "instruction": "<|im_start|>system\n",
                "input": "<|im_start|>user\n",
                "response": "<|im_start|>assistant\n",
                "endtoken": "<|im_end|>\n",
            },
            "llama3": {
                "instruction": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n",
                "input": "<|start_header_id|>user<|end_header_id|>\n",
                "response": "<|start_header_id|>assistant<|end_header_id|>\n",
                "endtoken": "<|eot_id|>",
            },
            "vicuna": {
                "instruction": "",
                "input": "USER:",
                "response": "ASSISTANT:",
                "endtoken": "",
            },
            "solar": {
                "instruction": "",
                "input": "<s> ### User:",
                "response": "### Assistant:",
                "endtoken": "",
            },
        }

        if model_type not in formats:
            msg = f"Unknown model type: {model_type}"
            raise UnknownModelTypeError(msg)

        return formats[model_type]

    def _warn_model_type_mismatch(self, model_path: str, model_type: str) -> None:
        model_path_lower = model_path.lower()
        model_type_lower = model_type.lower()
        hints: list[tuple[str, str]] = [
            ("mistral", "mistral"),
            ("llama-3", "llama3"),
            ("llama3", "llama3"),
        ]
        for hint, expected in hints:
            if hint in model_path_lower and model_type_lower != expected:
                logger.warning(
                    "Model path suggests '{}' but MODEL_TYPE is '{}'.",
                    expected,
                    model_type,
                )
                return

    def _read_llama_ctx_value(self, client: object, attr_name: str) -> int | None:
        value = getattr(client, attr_name, None)
        if callable(value):
            try:
                return int(value())
            except (TypeError, ValueError):
                return None
        if isinstance(value, int):
            return value
        return None

    def _check_model_context(self, model: LlamaCpp, configured_n_ctx: int | None) -> None:
        if not self.runtime_config.check_model_context:
            return
        client = getattr(model, "client", None)
        if client is None:
            return
        trained_ctx = self._read_llama_ctx_value(client, "n_ctx_train")
        active_ctx = self._read_llama_ctx_value(client, "n_ctx")
        if trained_ctx is None or active_ctx is None:
            return
        if active_ctx < trained_ctx:
            logger.warning(
                "Model trained context ({}) is larger than active context ({}).",
                trained_ctx,
                active_ctx,
            )
        if configured_n_ctx is None:
            logger.debug("No N_CTX set; active context is {} (trained {}).", active_ctx, trained_ctx)

    def _maybe_adjust_model_context(
        self,
        model: LlamaCpp,
        configured_n_ctx: int | None,
        llm_kwargs: dict[str, object],
    ) -> LlamaCpp:
        trained_ctx = None
        active_ctx = None
        client = getattr(model, "client", None)
        if client is not None:
            trained_ctx = self._read_llama_ctx_value(client, "n_ctx_train")
            active_ctx = self._read_llama_ctx_value(client, "n_ctx")

        if not self.runtime_config.auto_adjust_model_context:
            self._check_model_context(model, configured_n_ctx)
            return model

        if trained_ctx is None or active_ctx is None:
            self._check_model_context(model, configured_n_ctx)
            return model

        if active_ctx >= trained_ctx:
            self._check_model_context(model, configured_n_ctx)
            return model

        logger.warning(
            "Active context ({}) is smaller than trained context ({}). Reloading with trained context.",
            active_ctx,
            trained_ctx,
        )
        with contextlib.suppress(Exception):
            model.__del__()
        llm_kwargs = {**llm_kwargs, "n_ctx": trained_ctx}
        reloaded_model = LlamaCpp(**llm_kwargs)
        self._check_model_context(reloaded_model, trained_ctx)
        return reloaded_model

    def _load_character_card(self) -> dict:
        """Load and parse the character card JSON file."""
        card_source = Path("./cards/") / self.configs["CHARACTER_CARD"]
        logger.debug("Loading card from: {}", card_source)
        with card_source.open() as f:
            card_file = f.read()
            card_json = json.loads(card_file)
        return card_json["data"]

    def _load_prompt_template(self) -> BasePromptTemplate:
        """Load the prompt template from JSON file."""
        prompt_template = Path("./configs/") / self.configs["PROMPT_TEMPLATE"]
        logger.debug("Loading prompt from: {}", prompt_template)
        return load_prompt(prompt_template)

    def parse_prompt(self) -> BasePromptTemplate:
        """Parse character card and create prompt template."""
        # Load card and template
        card = self._load_character_card()
        prompt = self._load_prompt_template()

        # Extract character name
        char_name = card["name"] if "name" in card else card["char_name"]
        self.character_name = char_name

        # Get model format tokens
        model_type = self.configs["MODEL_TYPE"]
        tokens = self._get_model_format_tokens(model_type)

        self.llama_instruction = tokens["instruction"]
        self.llama_input = tokens["input"]
        self.llama_response = tokens["response"]
        self.llama_endtoken = tokens["endtoken"]

        if self.llama_instruction.startswith("<s>"):
            self.llama_instruction = self.llama_instruction.removeprefix("<s>")
            self.add_bos = True

        # Extract and process card fields
        description = card.get("description") or card.get("char_persona", "")
        scenario = card.get("scenario") or card.get("world_scenario", "")
        mes_example = card.get("mes_example") or card.get("example_dialogue", "")
        first_message = card.get("first_mes") or card.get("char_greeting", "")
        voice_instructions = card.get("voice_instructions", "")

        description = self._replace_template_variables(description, char_name=char_name)
        scenario = self._replace_template_variables(scenario, char_name=char_name)
        mes_example = self._replace_template_variables(mes_example, char_name=char_name)
        first_message = self._replace_template_variables(first_message, char_name=char_name)

        self.description = description
        self.scenario = scenario
        self.mes_example = mes_example
        self.first_message = first_message
        self.voice_instructions = voice_instructions

        return prompt.partial(
            character=char_name,
            description=description,
            scenario=scenario,
            mes_example=mes_example,
            llama_input=self.llama_input,
            llama_instruction=self.llama_instruction,
            llama_response=self.llama_response,
            llama_endtoken=self.llama_endtoken,
            vector_context=" ",
        )

    def instantiate_llm(self) -> LlamaCpp:
        """Instantiate and configure the LlamaCpp model."""
        model_path = Path("./models/") / self.configs["MODEL"]
        model = str(model_path)
        if not model:
            msg = "Could not load the model"
            raise ModelLoadError(msg)

        self._warn_model_type_mismatch(model, self.configs["MODEL_TYPE"])
        logger.debug("Loading model from: {}", model)
        # Add things here if you want to play with the model params
        # MAX_TOKENS is an optional param for when model answer cuts off
        # This can happen when large context models are told to print multiple paragraphs
        # Setting MAX_TOKENS lower than the context size can sometimes fix this

        stop_sequences = None
        model_type = self.runtime_config.model_type.lower()
        add_bos = self.add_bos
        if model_type == "mistral":
            stop_sequences = ["\nUser:", "User:", "\nUSER:", "USER:", "\n---", "\n==="]
            if self.character_name:
                stop_sequences.append(f"\n{self.character_name}:")
            add_bos = True

        model_kwargs = {"add_bos": add_bos}
        configured_n_ctx = self.configs.get("N_CTX")
        if configured_n_ctx is not None:
            configured_n_ctx = int(configured_n_ctx)

        # Calculate optimal GPU layers (supports "auto" or integer values)
        target_vram_usage = self.runtime_config.target_vram_usage
        n_gpu_layers = get_n_gpu_layers(
            model_path=model_path,
            configured_layers=self.runtime_config.layers,
            n_ctx=configured_n_ctx or 2048,
            target_vram_usage=target_vram_usage,
        )

        # Get KV cache quantization setting (q8_0, q4_0, or f16 for full precision)
        kv_quant = self.runtime_config.kv_cache_quant
        if kv_quant not in ("f16", "q8_0", "q4_0"):
            logger.warning("Invalid KV_CACHE_QUANT value '{}', using f16", kv_quant)
            kv_quant = "f16"

        configured_max_tokens = int(self.configs["MAX_TOKENS"])
        hard_max_tokens = self.configs.get("HARD_MAX_TOKENS")
        if hard_max_tokens is not None:
            configured_max_tokens = min(configured_max_tokens, int(hard_max_tokens))

        llm_kwargs = {
            "model_path": model,
            "streaming": True,
            "last_n_tokens_size": int(self.configs["LAST_N_TOKENS_SIZE"]),
            "n_batch": int(self.configs["N_BATCH"]),
            "max_tokens": configured_max_tokens,
            "use_mmap": False,
            "top_p": float(self.configs["TOP_P"]),
            "top_k": int(self.configs["TOP_K"]),
            "temperature": float(self.configs["TEMPERATURE"]),
            "repeat_penalty": float(self.configs["REPEAT_PENALTY"]),
            "n_gpu_layers": n_gpu_layers,
            "rope_freq_scale": 1,
            "model_kwargs": model_kwargs,
            "stop": stop_sequences,
            "verbose": False,
        }

        # Add KV cache quantization if not full precision
        if kv_quant != "f16":
            llm_kwargs["kvn_type"] = kv_quant  # llama-cpp-python supports kvn_type
            logger.debug("KV cache quantization enabled: {}", kv_quant)
        if configured_n_ctx is not None:
            llm_kwargs["n_ctx"] = configured_n_ctx

        llm_model = LlamaCpp(**llm_kwargs)
        return self._maybe_adjust_model_context(llm_model, configured_n_ctx, llm_kwargs)

    def _get_vector_client(self) -> chromadb.PersistentClient:
        if self._vector_client is None:
            self._vector_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._vector_client

    def _get_vector_embedder(self) -> HuggingFaceEmbeddings:
        if self._vector_embedder is None:
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": True}
            self._vector_embedder = HuggingFaceEmbeddings(
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                cache_folder=self.embedding_cache,
            )
        return self._vector_embedder

    def _get_vector_db(self, collection_name: str) -> Chroma:
        if collection_name not in self._vector_dbs:
            self._vector_dbs[collection_name] = Chroma(
                client=self._get_vector_client(),
                collection_name=collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self._get_vector_embedder(),
            )
        return self._vector_dbs[collection_name]

    def _get_cross_encoder(self) -> CrossEncoder:
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(self.runtime_config.rag_rerank_model, device="cpu")
        return self._cross_encoder

    @staticmethod
    def _describe_where_filter(where: WhereFilter) -> str:
        if where is None:
            return "unfiltered"
        if "$and" in where:
            return "$and"
        if "$or" in where:
            return "$or"
        return "metadata"

    def _rerank_chunks(self, query: str, chunks: list[str], k: int) -> list[str]:
        if not chunks:
            return []

        top_n = max(k, self.runtime_config.rag_rerank_top_n)
        candidates = chunks[:top_n]
        if len(candidates) <= 1:
            return chunks[:k]

        try:
            cross_encoder = self._get_cross_encoder()
            pairs = [(query, chunk) for chunk in candidates]
            scores = cross_encoder.predict(pairs, show_progress_bar=False)
            scored_chunks = list(zip(candidates, scores, strict=False))
            scored_chunks.sort(key=lambda item: float(item[1]), reverse=True)
            reranked = [chunk for chunk, _score in scored_chunks]
            if len(chunks) > len(candidates):
                reranked.extend(chunks[len(candidates) :])
            return reranked[:k]
        except Exception as error:
            logger.warning("Reranking failed; using first-pass retrieval order. Error: {}", error)
            return chunks[:k]

    @staticmethod
    def _score_spread(scores: list[float]) -> tuple[float | None, float | None, float | None]:
        if not scores:
            return None, None, None
        min_score = min(scores)
        max_score = max(scores)
        return min_score, max_score, max_score - min_score

    @staticmethod
    def _run_mmr_search(
        db: Chroma,
        query: str,
        where: WhereFilter,
        search_options: dict[str, object],
    ) -> list[str]:
        search_kwargs: dict[str, object] = {
            "query": query,
            "k": int(search_options["retrieval_k"]),
            "fetch_k": int(search_options["fetch_k"]),
            "lambda_mult": float(search_options["lambda_mult"]),
        }
        if where is not None:
            search_kwargs["filter"] = where
        docs = db.max_marginal_relevance_search(**search_kwargs)
        return [doc.page_content for doc in docs]

    @staticmethod
    def _run_similarity_search(
        db: Chroma,
        query: str,
        where: WhereFilter,
        retrieval_k: int,
        score_threshold: object,
    ) -> tuple[list[str], list[float]]:
        search_kwargs: dict[str, object] = {"query": query, "k": retrieval_k}
        if where is not None:
            search_kwargs["filter"] = where

        docs = db.similarity_search_with_score(**search_kwargs)
        if score_threshold is not None:
            threshold_value = float(score_threshold)
            docs = [(doc, score) for doc, score in docs if score <= threshold_value]

        chunks = [doc.page_content for doc, _score in docs]
        scores = [float(score) for _doc, score in docs]
        return chunks, scores

    def _log_retrieval_telemetry(
        self,
        query: str,
        main_trace: RetrievalTrace,
        mes_trace: RetrievalTrace,
        cleanup_stats: dict[str, int],
    ) -> None:
        if not self.runtime_config.rag_telemetry_enabled:
            return

        logger.info(
            "RAG telemetry: query_chars={} main(mode={} filter={} cand={} out={} spread={}) "
            "mes(mode={} filter={} cand={} out={} spread={}) cleanup(main={} mes={} cross_removed={})",
            len(query),
            main_trace.get("mode", "unknown"),
            main_trace.get("filter_path", "n/a"),
            main_trace.get("candidates", 0),
            main_trace.get("returned", 0),
            main_trace.get("score_spread", "n/a"),
            mes_trace.get("mode", "unknown"),
            mes_trace.get("filter_path", "n/a"),
            mes_trace.get("candidates", 0),
            mes_trace.get("returned", 0),
            mes_trace.get("score_spread", "n/a"),
            cleanup_stats.get("main", 0),
            cleanup_stats.get("mes", 0),
            cleanup_stats.get("cross_removed", 0),
        )

    def _dedupe_cross_collection_chunks(
        self, context_chunks: list[str], mes_chunks: list[str]
    ) -> tuple[list[str], list[str], int]:
        if not context_chunks or not mes_chunks:
            return context_chunks, mes_chunks, 0

        seen_signatures = {self._chunk_signature(chunk) for chunk in context_chunks}
        deduped_mes: list[str] = []
        removed = 0

        for chunk in mes_chunks:
            signature = self._chunk_signature(chunk)
            if signature in seen_signatures:
                removed += 1
                continue
            seen_signatures.add(signature)
            deduped_mes.append(chunk)

        return context_chunks, deduped_mes, removed

    def _search_collection_with_trace(
        self,
        collection_name: str,
        query: str,
        filters: list[WhereFilter],
        k: int | None = None,
    ) -> tuple[list[str], RetrievalTrace]:
        trace: RetrievalTrace = {
            "mode": "mmr" if self.runtime_config.use_mmr else "similarity",
            "filter_path": "none",
            "candidates": 0,
            "returned": 0,
            "score_spread": None,
            "rerank_applied": False,
        }

        if not query:
            return [], trace

        if k is None:
            k = self.rag_k

        retrieval_k = max(k, self.runtime_config.rag_rerank_top_n) if self.runtime_config.rag_rerank_enabled else k
        db = self._get_vector_db(collection_name)
        use_mmr = self.runtime_config.use_mmr
        fetch_k = self.runtime_config.rag_fetch_k
        lambda_mult = self.runtime_config.lambda_mult
        score_threshold = self.configs.get("RAG_SCORE_THRESHOLD")
        candidate_chunks: list[str] = []
        similarity_scores: list[float] = []

        for where in filters:
            trace["filter_path"] = self._describe_where_filter(where)
            if use_mmr:
                candidate_chunks = self._run_mmr_search(
                    db,
                    query,
                    where,
                    {
                        "retrieval_k": retrieval_k,
                        "fetch_k": max(retrieval_k, fetch_k),
                        "lambda_mult": lambda_mult,
                    },
                )
                if candidate_chunks or where is None:
                    break
                continue

            candidate_chunks, similarity_scores = self._run_similarity_search(
                db,
                query,
                where,
                retrieval_k,
                score_threshold,
            )
            if candidate_chunks or where is None:
                break

        result_chunks = candidate_chunks[:k]
        if self.runtime_config.rag_rerank_enabled and candidate_chunks:
            result_chunks = self._rerank_chunks(query=query, chunks=candidate_chunks, k=k)
            trace["rerank_applied"] = True

        score_min, score_max, score_spread = self._score_spread(similarity_scores)
        trace["candidates"] = len(candidate_chunks)
        trace["returned"] = len(result_chunks)
        trace["score_min"] = score_min
        trace["score_max"] = score_max
        trace["score_spread"] = score_spread

        return result_chunks, trace

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

    def _is_low_quality_context_chunk(self, chunk: str) -> bool:
        text = chunk.strip()
        if not text:
            return True
        return any(pattern.search(text) for pattern in self._LOW_QUALITY_CONTEXT_PATTERNS)

    def _filter_context_chunks(self, chunks: list[str]) -> list[str]:
        """Remove low-quality boilerplate chunks and exact duplicates while preserving order."""
        filtered: list[str] = []
        seen: set[str] = set()
        seen_signatures: set[str] = set()
        for chunk in chunks:
            normalized = chunk.strip()
            if not normalized:
                continue
            normalized = self._dedupe_chunk_sections(normalized)
            if not normalized:
                continue
            if normalized in seen:
                continue
            if self._is_low_quality_context_chunk(normalized):
                continue
            signature = self._chunk_signature(normalized)
            if signature in seen_signatures:
                continue
            seen.add(normalized)
            seen_signatures.add(signature)
            filtered.append(normalized)
        return filtered

    def _dedupe_chunk_sections(self, chunk: str) -> str:
        """Remove repeated markdown sections inside a single chunk."""
        lines = chunk.splitlines()
        if not lines:
            return chunk

        output_lines: list[str] = []
        current_section: list[str] = []
        seen_section_titles: set[str] = set()
        current_title: str | None = None

        def flush_section() -> None:
            nonlocal current_section, current_title
            if not current_section:
                return
            if current_title is None:
                output_lines.extend(current_section)
            else:
                title_key = current_title.lower().strip()
                if title_key not in seen_section_titles:
                    seen_section_titles.add(title_key)
                    output_lines.extend(current_section)
            current_section = []
            current_title = None

        for line in lines:
            heading_match = self._MARKDOWN_HEADING_RE.match(line.strip())
            if heading_match:
                flush_section()
                current_title = heading_match.group(1)
                current_section = [line]
            elif current_section:
                current_section.append(line)
            else:
                output_lines.append(line)

        flush_section()

        return "\n".join(output_lines).strip()

    def _chunk_signature(self, chunk: str) -> str:
        """Build a lightweight signature for near-duplicate chunk elimination."""
        signature_lines: list[str] = []
        for line in chunk.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Ignore heading markers in signature so equivalent sections with
            # minor heading level differences collapse together.
            stripped = self._MARKDOWN_HEADING_RE.sub(r"\\1", stripped)
            stripped = re.sub(r"\s+", " ", stripped).lower()
            signature_lines.append(stripped)
            if len(signature_lines) >= self._CHUNK_SIGNATURE_LINES:
                break
        return "|".join(signature_lines)

    def _cap_context_text(self, text: str) -> str:
        """Cap vector context length to avoid prompt bloat while preserving boundaries."""
        max_chars = self.runtime_config.max_vector_context_chars
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        clipped = text[:max_chars]
        if "\n\n" in clipped:
            clipped = clipped.rsplit("\n\n", 1)[0]
        elif "\n" in clipped:
            clipped = clipped.rsplit("\n", 1)[0]
        return clipped.strip()

    def _should_skip_rag_for_message(self, message: str) -> bool:
        """Return True for short small-talk turns where lore retrieval hurts quality."""
        normalized = re.sub(r"\s+", " ", message).strip()
        if not normalized:
            return True
        words = [word for word in normalized.split(" ") if word]
        max_words = self.runtime_config.small_talk_max_words
        if len(words) > max_words:
            return False
        return any(pattern.search(normalized) for pattern in self._SMALL_TALK_PATTERNS)

    def _should_skip_rag_for_followup(self, message: str) -> bool:
        """Skip RAG on short follow-up chat turns without lore/entity matches."""
        normalized = re.sub(r"\s+", " ", message).strip()
        if not normalized:
            return True
        max_words = self.runtime_config.followup_rag_max_words
        words = [word for word in normalized.split(" ") if word]
        if len(words) > max_words:
            return False
        if not self.rag_collection:
            return True
        matches = self._get_key_matches(normalized, self.rag_collection)
        return len(matches) == 0

    def _search_collection(
        self,
        collection_name: str,
        query: str,
        filters: list[WhereFilter],
        k: int | None = None,
    ) -> list[str]:
        chunks, _trace = self._search_collection_with_trace(collection_name, query, filters, k=k)
        return chunks

    def _query_terms(self, query: str) -> set[str]:
        terms = re.findall(r"[a-zA-Z0-9_]+", query.lower())
        return {term for term in terms if len(term) > self._MIN_QUERY_TERM_LEN and term not in self._QUERY_STOPWORDS}

    def _build_multi_queries(self, query: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", query).strip()
        if not normalized:
            return []

        max_variants = max(1, self.runtime_config.rag_multi_query_max_variants)
        variants: list[str] = [normalized]
        terms = sorted(self._query_terms(normalized))

        # Lexical reformulations that preserve intent while giving retrieval multiple angles.
        compact_query_term_count = self._COMPACT_QUERY_TERM_COUNT
        if len(terms) >= compact_query_term_count:
            variants.append(" ".join(terms[:compact_query_term_count]))
            variants.append(" ".join(terms[-compact_query_term_count:]))
        elif terms:
            variants.append(" ".join(terms))

        # Keep deterministic order and uniqueness.
        deduped: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            cleaned = variant.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
            if len(deduped) >= max_variants:
                break
        return deduped

    def _merge_multi_query_chunks(self, ranked_results: list[list[str]], k: int) -> list[str]:
        if not ranked_results:
            return []

        merged: list[str] = []
        seen_signatures: set[str] = set()
        max_depth = max(len(result) for result in ranked_results)

        for index in range(max_depth):
            for result in ranked_results:
                if index >= len(result):
                    continue
                chunk = result[index].strip()
                if not chunk:
                    continue
                signature = self._chunk_signature(chunk)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                merged.append(chunk)
                if len(merged) >= k:
                    return merged
        return merged

    def _multi_query_search_with_trace(
        self,
        collection_name: str,
        base_query: str,
        filters: list[WhereFilter],
        k: int,
    ) -> tuple[list[str], RetrievalTrace]:
        queries = [base_query]
        if self.runtime_config.rag_multi_query_enabled:
            queries = self._build_multi_queries(base_query)

        if not queries:
            return [], {
                "mode": "mmr" if self.runtime_config.use_mmr else "similarity",
                "filter_path": "none",
                "candidates": 0,
                "returned": 0,
                "score_spread": None,
                "rerank_applied": False,
                "queries": 0,
            }

        traces: list[RetrievalTrace] = []
        ranked_results: list[list[str]] = []
        for retrieval_query in queries:
            chunks, trace = self._search_collection_with_trace(collection_name, retrieval_query, filters, k=k)
            traces.append(trace)
            ranked_results.append(chunks)

        merged_chunks = self._merge_multi_query_chunks(ranked_results, k)
        aggregate_trace = traces[0] if traces else {}
        aggregate_trace = {
            **aggregate_trace,
            "queries": len(queries),
            "candidates": sum(int(trace.get("candidates", 0)) for trace in traces),
            "returned": len(merged_chunks),
            "multi_query_enabled": self.runtime_config.rag_multi_query_enabled,
        }
        return merged_chunks, aggregate_trace

    def _score_context_sentences(self, query_terms: set[str], context: str) -> list[tuple[int, int, str]]:
        blocks = [block.strip() for block in context.split("\n\n") if block.strip()]
        scored_sentences: list[tuple[int, int, str]] = []
        sentence_position = 0

        for block in blocks:
            normalized_block = re.sub(r"\s+", " ", block).strip()
            if not normalized_block:
                continue

            sentences = [s.strip() for s in self._SENTENCE_SPLIT_RE.split(normalized_block) if s.strip()]
            for sentence in sentences:
                sentence_terms = self._query_terms(sentence)
                overlap = len(query_terms.intersection(sentence_terms))
                if overlap > 0:
                    scored_sentences.append((overlap, -sentence_position, sentence))
                sentence_position += 1

        scored_sentences.sort(reverse=True)
        return scored_sentences

    def _compress_context_sentences(self, query: str, context: str) -> str:
        if not context or not self.runtime_config.rag_sentence_compression_enabled:
            return context

        max_sentences = self.runtime_config.rag_sentence_compression_max_sentences
        if max_sentences <= 0:
            return context

        query_terms = self._query_terms(query)
        if not query_terms:
            return context

        scored_sentences = self._score_context_sentences(query_terms, context)
        if not scored_sentences:
            return context

        selected: list[str] = []
        seen_sentences: set[str] = set()
        for _score, _position, sentence in scored_sentences:
            sentence_key = sentence.lower()
            if sentence_key in seen_sentences:
                continue
            seen_sentences.add(sentence_key)
            selected.append(sentence)
            if len(selected) >= max_sentences:
                break

        return "\n\n".join(selected) if selected else context

    def _get_vector_context(self, query: str, k: int | None = None) -> tuple[str, str]:
        if not self.rag_collection:
            return "", ""
        # Enrich query with character name to orient embedding search toward the character domain
        enriched_query = f"{self.character_name} {query}" if self.character_name else query
        matches = self._get_key_matches(query, self.rag_collection)
        filters = build_where_filters(matches)
        retrieval_k = k if k is not None else self.rag_k
        context_chunks, context_trace = self._multi_query_search_with_trace(
            self.rag_collection, enriched_query, filters, k=retrieval_k
        )
        # Use unfiltered search for message examples: goal is stylistic match, not factual (Section 6.2)
        k_mes = k if k is not None else self.rag_k_mes
        mes_chunks, mes_trace = self._multi_query_search_with_trace(
            f"{self.rag_collection}_mes",
            enriched_query,
            [None],
            k=k_mes,
        )
        context_chunks = self._filter_context_chunks(context_chunks)
        mes_chunks = self._filter_context_chunks(mes_chunks)
        context_chunks, mes_chunks, cross_removed = self._dedupe_cross_collection_chunks(context_chunks, mes_chunks)
        self._log_retrieval_telemetry(
            query=enriched_query,
            main_trace=context_trace,
            mes_trace=mes_trace,
            cleanup_stats={"main": len(context_chunks), "mes": len(mes_chunks), "cross_removed": cross_removed},
        )
        vector_context = "\n\n".join(context_chunks)
        vector_context = self._dedupe_chunk_sections(vector_context)
        vector_context = self._cap_context_text(vector_context)
        vector_context = self._compress_context_sentences(query=enriched_query, context=vector_context)
        vector_context = self._cap_context_text(vector_context)
        mes_example = "\n\n".join(mes_chunks)
        return vector_context, mes_example

    def get_history(self) -> str:
        """Build conversation history from message deques."""
        summary_block = self._get_history_summary_block()
        message_history = ""
        user_message_history_list = list(self.user_message_history)
        ai_message_history_list = list(self.ai_message_history)
        if len(user_message_history_list) != len(ai_message_history_list):
            msg = "User and AI message history are out of sync"
            raise ValueError(msg)

        for x in range(len(user_message_history_list)):
            user_message = user_message_history_list[x]
            ai_message = ai_message_history_list[x]
            new_line = f"User: {user_message}\n{self.character_name}:{ai_message}\n"
            message_history = message_history + new_line
        return f"{summary_block}{message_history}" if summary_block else message_history

    def _get_history_summary_block(self) -> str:
        if not self.history_summaries:
            return ""
        summary_lines = "\n".join(self.history_summaries)
        return "[Conversation summary of earlier turns]\n" + summary_lines + "\n\n"

    def _clip_summary_text(self, text: str) -> str:
        max_chars = max(40, self.runtime_config.history_summarization_max_chars)
        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) <= max_chars:
            return normalized
        clipped = normalized[:max_chars].rsplit(" ", 1)[0]
        return f"{clipped.rstrip('.!?')}." if clipped else normalized[:max_chars]

    def _topic_label(self, terms: set[str]) -> str:
        if not terms:
            return "general"
        return ", ".join(sorted(terms)[:3])

    def _is_topic_shift(self, previous_terms: set[str], current_terms: set[str]) -> bool:
        if not previous_terms or not current_terms:
            return False
        overlap = len(previous_terms.intersection(current_terms))
        return overlap == 0

    def _build_summary_entry(self, user_message: str, ai_message: str) -> str:
        user_terms = self._query_terms(user_message)
        topic_shift_note = ""
        if self._is_topic_shift(self._last_summary_topic_terms, user_terms):
            from_topic = self._topic_label(self._last_summary_topic_terms)
            to_topic = self._topic_label(user_terms)
            topic_shift_note = f" [Topic shift: {from_topic} -> {to_topic}]"

        user_text = self._clip_summary_text(user_message)
        ai_text = self._clip_summary_text(ai_message)
        if user_terms:
            self._last_summary_topic_terms = user_terms
        topic_label = self._topic_label(user_terms)
        return (
            f"- User asked about {topic_label}: {user_text} | "
            f"{self.character_name}: {ai_text}{topic_shift_note}"
        )

    def _compact_history_if_needed(self) -> None:
        if not self.runtime_config.history_summarization_enabled:
            return
        threshold = max(2, self.runtime_config.history_summarization_threshold)
        keep_recent = max(1, self.runtime_config.history_summarization_keep_recent)
        keep_recent = min(keep_recent, len(self.user_message_history))

        while len(self.user_message_history) > keep_recent and len(self.user_message_history) >= threshold:
            user_message = self.user_message_history.popleft()
            ai_message = self.ai_message_history.popleft()
            summary_entry = self._build_summary_entry(user_message, ai_message)
            self.history_summaries.append(summary_entry)

    def _build_mistral_prompt(self, message: str, vector_context: str, mes_example: str) -> str:
        # Build base system prompt WITHOUT vector context (it goes in final block only)
        voice_section = f"\n{self.voice_instructions}" if self.voice_instructions else ""
        summary_block = self._get_history_summary_block()
        summary_section = f"\n\nEarlier Conversation Summary:\n{summary_block}" if summary_block else ""
        system_prompt = (
            f"You are roleplaying as {self.character_name} in a continuous fictional chat with User. "
            "Stay in character, follow the description and scenario, and use the examples "
            f"and context as guidance.{voice_section}\n\n"
            f"{self.description}\n\n"
            f"Scenario:\n{self.scenario}\n\n"
            f"Message Examples:\n{mes_example}\n\n"
            f"{summary_section}\n\n"
            "Do not repeat previous responses verbatim. Do not narrate static scene descriptions unless asked.\n\n"
            f"Reply only as {self.character_name}; never write any User lines "
            "(for example, never include 'User:' in your output) or dialogue for User."
        ).strip()

        user_message_history_list = list(self.user_message_history)
        ai_message_history_list = list(self.ai_message_history)
        if len(user_message_history_list) != len(ai_message_history_list):
            msg = "User and AI message history are out of sync"
            raise ValueError(msg)

        blocks: list[str] = []
        for idx, (user_message, ai_message) in enumerate(
            zip(user_message_history_list, ai_message_history_list, strict=True),
        ):
            inst_content = f"{system_prompt}\n\nUser: {user_message}" if idx == 0 else f"User: {user_message}"
            blocks.append(f"<s>[INST] {inst_content} [/INST] {ai_message}</s>")

        # Only include vector_context in the final current message block
        current_content = f"User: {message}" if blocks else f"{system_prompt}\n\nUser: {message}"
        if vector_context and vector_context.strip():
            current_content = f"{vector_context}\n\n{current_content}"
        blocks.append(f"<s>[INST] {current_content} [/INST]")
        return "\n".join(blocks)

    def _build_system_prompt_text(self, mes_example: str) -> str:
        """Build the system prompt text without vector context."""
        voice_section = f"\n{self.voice_instructions}" if self.voice_instructions else ""
        summary_block = self._get_history_summary_block()
        summary_section = f"\n\nEarlier Conversation Summary:\n{summary_block}" if summary_block else ""
        return (
            f"You are roleplaying as {self.character_name} in a continuous fictional chat with User. "
            "Stay in character, follow the description and scenario, and use the examples "
            f"and context as guidance.{voice_section}\n\n"
            f"{self.description}\n\n"
            f"Scenario:\n{self.scenario}\n\n"
            f"Message Examples:\n{mes_example}\n\n"
            f"{summary_section}\n\n"
            "Do not repeat previous responses verbatim. Do not narrate static scene descriptions unless asked.\n\n"
            f"Reply only as {self.character_name}; never write any User lines "
            "(for example, never include 'User:' in your output) or dialogue for User."
        ).strip()

    def _prepare_static_vector_context(self, message: str, mes_example: str) -> tuple[str, str]:
        vector_context, mes_from_rag = self._get_vector_context(message)
        if mes_from_rag:
            mes_example = f"{mes_example}\n\n{mes_from_rag}".strip() if mes_example else mes_from_rag
        elif not mes_example:
            mes_example = self.mes_example
        return vector_context, mes_example

    def _prepare_dynamic_vector_context(self, message: str, is_first_turn: bool, mes_example: str) -> tuple[str, str]:
        system_prompt = self._build_system_prompt_text(self.mes_example if is_first_turn else "")
        budget = self.context_manager.calculate_budget(system_prompt)

        if budget.budget_for_dynamic_content < self._MIN_DYNAMIC_CONTENT_TOKENS:
            logger.warning(
                "Available budget ({} tokens) too small for dynamic allocation. Using static mode.",
                budget.budget_for_dynamic_content,
            )
            return self._prepare_static_vector_context(message, mes_example)

        chunk_size_estimate = self.runtime_config.chunk_size_estimate
        context_budget_estimate = budget.budget_for_dynamic_content * 0.25
        initial_k = max(self.rag_k, int(context_budget_estimate / chunk_size_estimate))
        max_initial = self.runtime_config.max_initial_retrieval
        initial_k = min(initial_k, max_initial)

        vector_context_full, _mes_from_rag_full = self._get_vector_context(message, k=initial_k)
        history = self.get_history()
        allocation = self.context_manager.allocate_content(
            budget,
            self.mes_example if is_first_turn else "",
            vector_context_full,
            history,
            message,
        )
        vector_context = allocation["allocated_context"]
        if is_first_turn:
            mes_example = allocation["allocated_examples"]
        else:
            mes_from_rag = allocation["allocated_examples"]
            if mes_from_rag:
                mes_example = mes_from_rag

        if self.runtime_config.debug_context:
            logger.debug(self.context_manager.get_context_info(budget, allocation))

        return vector_context, mes_example

    def _prepare_vector_context(self, message: str) -> tuple[str, str]:
        """Prepare vector context with optional dynamic allocation."""
        vector_context = ""
        is_first_turn = not self.user_message_history and not self.ai_message_history
        mes_example = self.mes_example if is_first_turn else ""

        if self._should_skip_rag_for_message(message):
            return " ", mes_example
        if (not is_first_turn) and self._should_skip_rag_for_followup(message):
            return " ", mes_example

        try:
            if self.use_dynamic_context and not is_first_turn:
                vector_context, mes_example = self._prepare_dynamic_vector_context(message, is_first_turn, mes_example)
            else:
                vector_context, mes_example = self._prepare_static_vector_context(message, mes_example)
        except Exception as e:
            logger.warning("Error in dynamic context allocation: {}. Using static fallback.", e)
            vector_context = ""
            if not mes_example:
                mes_example = self.mes_example

        vector_context = (
            (
                "[The following background information is relevant to the current topic. "
                "Use it to inform your response but do not quote it directly.]\n"
                f"{vector_context}"
            )
            if vector_context
            else " "
        )
        return vector_context, mes_example

    def _build_conversation_chain(self, message: str, vector_context: str, mes_example: str) -> tuple[object, object]:
        model_type = self.runtime_config.model_type.lower()
        output_parser = StrOutputParser()
        if model_type == "mistral":
            prompt_text = self._build_mistral_prompt(message, vector_context, mes_example)
            if prompt_text.startswith("<s>"):
                prompt_text = prompt_text.removeprefix("<s>")
            if self.runtime_config.debug_prompt:
                logger.debug("Mistral prompt:\n{}", prompt_text)
            conversation_chain = self.llm_model | output_parser
            chain_input = prompt_text
        else:
            history = self.get_history()
            query_input = {
                "input": message,
                "history": history,
                "vector_context": vector_context,
                "mes_example": mes_example,
            }
            if self.runtime_config.debug_prompt:
                logger.debug("Prompt template input: {}", query_input)
            conversation_chain = self.prompt | self.llm_model | output_parser
            chain_input = query_input
        self._maybe_log_prompt_fingerprint(chain_input)
        return conversation_chain, chain_input

    def _maybe_log_prompt_fingerprint(self, chain_input: object) -> None:
        """Log deterministic prompt fingerprints for cross-entrypoint comparisons."""
        if not self.runtime_config.debug_prompt_fingerprint:
            return
        payload = chain_input if isinstance(chain_input, str) else json.dumps(chain_input, sort_keys=True)
        prompt_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        history_turns = min(len(self.user_message_history), len(self.ai_message_history))
        logger.debug(
            "Prompt fingerprint: hash={} model_type={} history_turns={} chars={}",
            prompt_hash,
            self.runtime_config.model_type,
            history_turns,
            len(payload),
        )

    async def _stream_response(  # noqa: PLR0912, PLR0915
        self,
        conversation_chain: object,
        chain_input: object,
        first_token_event: threading.Event | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> str | None:
        chunks: list[str] = []
        raw_stream = ""
        max_stream_chars = self.runtime_config.max_stream_chars
        max_silent_stream_chars = self.runtime_config.max_silent_stream_chars
        empty_stream_fallback = self.runtime_config.empty_stream_fallback
        stop_stream = False
        visible_output_emitted = False
        silent_stream_chars = 0

        def emit_text(text: str) -> None:
            nonlocal visible_output_emitted
            if not text:
                return
            if not visible_output_emitted:
                text = text.lstrip()
                if not text:
                    return
            if text.strip():
                visible_output_emitted = True
            if stream_callback:
                stream_callback(text)
            else:
                print(text, flush=True, end="")  # noqa: T201

        def emit_fallback_response() -> str:
            emit_text(empty_stream_fallback)
            return empty_stream_fallback

        try:
            async for chunk in conversation_chain.astream(
                chain_input,
            ):
                chunk_str = str(chunk)
                if first_token_event is not None and not first_token_event.is_set():
                    first_token_event.set()
                chunks.append(chunk_str)
                raw_stream += chunk_str

                if not stop_stream and any(pattern in raw_stream for pattern in self._USER_TURN_PATTERNS):
                    logger.warning("Stopping stream early after detecting generated User-turn pattern.")
                    stop_stream = True
                if not stop_stream and max_stream_chars > 0 and len(raw_stream) >= max_stream_chars:
                    logger.warning(
                        "Stopping stream early after reaching MAX_STREAM_CHARS={}.",
                        max_stream_chars,
                    )
                    stop_stream = True

                if not visible_output_emitted and not chunk_str.strip():
                    silent_stream_chars += len(chunk_str)
                    if (
                        not stop_stream
                        and max_silent_stream_chars > 0
                        and silent_stream_chars >= max_silent_stream_chars
                    ):
                        logger.warning(
                            "Stopping stream early after receiving {} silent chars without visible output.",
                            silent_stream_chars,
                        )
                        stop_stream = True

                emit_text(chunk_str)

                if stop_stream:
                    break
        except (KeyboardInterrupt, asyncio.CancelledError):
            if first_token_event is not None:
                first_token_event.set()
            if not stream_callback:
                print()  # noqa: T201
            return None
        except Exception:
            # Suppress any underlying exceptions from llama_cpp during cleanup
            if first_token_event is not None:
                first_token_event.set()
            logger.warning("Streaming failed; emitting fallback response.")
            return emit_fallback_response()

        if not visible_output_emitted:
            return emit_fallback_response()

        return "".join(chunks)

    def update_history(self, message: str, result: str) -> None:
        """Update message history with user query and AI response."""
        if (
            self.runtime_config.history_summarization_enabled
            and self.user_message_history.maxlen is not None
            and len(self.user_message_history) >= self.user_message_history.maxlen
        ):
            user_message = self.user_message_history.popleft()
            ai_message = self.ai_message_history.popleft()
            self.history_summaries.append(self._build_summary_entry(user_message, ai_message))
        self.user_message_history.append(message)
        self.ai_message_history.append(result)
        self._compact_history_if_needed()

    _STRAY_TOKENS: tuple[str, ...] = ("[/INST]", "<|im_end|>", "</s>", "<|eot_id|>", "<s>", "<|end|>")
    # Patterns that identify a generated User turn. Newline-prefixed variants are used
    # for truncation (5.1); bare variants are used for quality gating (2.4) to catch
    # patterns that may appear without a leading newline after post-processing.
    _USER_TURN_PATTERNS: tuple[str, ...] = ("\nUser:", "\nUSER:", "\n{{user}}", "\nuser:")
    _USER_TURN_BASE_PATTERNS: tuple[str, ...] = ("User:", "USER:", "{{user}}", "user:")
    _MIN_RESPONSE_CHARS: int = 40  # approx 10 tokens; responses shorter than this are rejected

    def _post_process_response(self, response: str) -> str:
        """Clean up raw model output before storing in conversation history.

        Note: this runs after `_stream_response` has already emitted all chunks to the
        caller, so it affects **history storage only** — not what was displayed during
        streaming.  Applies user-turn truncation (5.1), stray-token removal, and
        whitespace normalisation (5.4).
        """
        # 5.1 Truncate at any generated User-turn pattern
        for pattern in self._USER_TURN_PATTERNS:
            if pattern in response:
                response = response[: response.index(pattern)]

        # 5.4 Remove stray model-format tokens
        for token in self._STRAY_TOKENS:
            response = response.replace(token, "")

        # 5.4 Collapse runs of 3+ newlines to a single blank line
        response = re.sub(r"\n{3,}", "\n\n", response)

        return response.strip()

    def _is_quality_response(self, response: str) -> bool:
        """Return True if a response passes basic quality checks (2.4).

        Rejects responses that are too short (< `_MIN_RESPONSE_CHARS` characters),
        break character by generating User-turn markers, or are exact duplicates of
        the previous AI turn.
        """
        if len(response.strip()) < self._MIN_RESPONSE_CHARS:
            logger.warning("Response too short ({} chars), skipping history.", len(response.strip()))
            return False

        # Check for broken-character User-turn patterns
        for pattern in self._USER_TURN_BASE_PATTERNS:
            if pattern in response:
                logger.warning("Response contains User-turn pattern '{}', skipping history.", pattern)
                return False

        # Reject exact duplicate of last AI turn
        if self.ai_message_history:
            last_ai = list(self.ai_message_history)[-1]
            if response.strip() == last_ai.strip():
                logger.warning("Response is identical to previous AI turn, skipping history.")
                return False

        return True

    async def ask_question(
        self,
        message: str,
        first_token_event: threading.Event | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> None:
        """Query the model with streaming output."""
        vector_context, mes_example = self._prepare_vector_context(message)
        conversation_chain, chain_input = self._build_conversation_chain(message, vector_context, mes_example)
        answer = await self._stream_response(conversation_chain, chain_input, first_token_event, stream_callback)
        if first_token_event is not None:
            first_token_event.set()
        if answer is None:
            return
        answer = self._post_process_response(answer)
        if self._is_quality_response(answer):
            self.update_history(message, answer)
        else:
            logger.warning("Response did not pass quality check; turn not added to history.")
            fallback_history_response = self.runtime_config.quality_fallback_response
            self.update_history(message, fallback_history_response)
