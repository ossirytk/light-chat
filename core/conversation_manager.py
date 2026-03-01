import asyncio
import contextlib
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

from core.collection_helper import build_where_filters, extract_key_matches, normalize_keyfile
from core.context_manager import ApproximateTokenCounter, ContextManager
from core.gpu_utils import get_n_gpu_layers


class UnknownModelTypeError(Exception):
    """Raised when an unsupported model type is configured."""


class ModelLoadError(Exception):
    """Raised when a model fails to load."""


class ConversationManager:
    _LOW_QUALITY_CONTEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\b(main antagonist of (the )?system shock series)\b", re.IGNORECASE),
        re.compile(r"\b(sits before user|stands before user|before user\.)\b", re.IGNORECASE),
        re.compile(r"\b(piercing green eyes|shimmering wires|cold demeanor)\b", re.IGNORECASE),
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
        self.prompt = self.parse_prompt()
        self.llm_model = self.instantiate_llm()
        self.persist_directory = self.configs.get("PERSIST_DIRECTORY", "./character_storage/")
        self.key_storage = self.configs.get("KEY_STORAGE", "./rag_data/")
        self.embedding_cache = self.configs.get("EMBEDDING_CACHE", "./embedding_models/")
        self.rag_k = int(self.configs.get("RAG_K", 7))
        self.rag_k_mes = int(self.configs.get("RAG_K_MES", self.rag_k))
        self.rag_collection = self.configs.get("RAG_COLLECTION", self._sanitize_collection_name(self.character_name))
        _default_max_history = 10
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
        self._vector_client: chromadb.PersistentClient | None = None
        self._vector_embedder: HuggingFaceEmbeddings | None = None
        self._vector_dbs: dict[str, Chroma] = {}

        # Initialize context manager for dynamic context window allocation
        self.context_manager = self._initialize_context_manager()
        self.use_dynamic_context = bool(self.configs.get("USE_DYNAMIC_CONTEXT", True))

    def __del__(self) -> None:
        """Cleanup the LlamaCpp model when the ConversationManager is destroyed."""
        if hasattr(self, "llm_model") and self.llm_model is not None:
            with contextlib.suppress(Exception):
                self.llm_model.__del__()

    def load_configs(self) -> dict:
        config_dir = Path("./configs/")
        model_config_source = config_dir / "modelconf.json"
        app_config_source = config_dir / "appconf.json"
        logger.debug("Loading model config from: {}", model_config_source)
        with model_config_source.open() as f:
            model_config = json.load(f)
        logger.debug("Loading app config from: {}", app_config_source)
        with app_config_source.open() as f:
            app_config = json.load(f)
        merged_config = {**model_config, **app_config}
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

        reserved_for_response = int(self.configs.get("RESERVED_FOR_RESPONSE", 256))

        return ContextManager(
            context_window=context_window,
            token_counter=ApproximateTokenCounter(),
            reserved_for_response=reserved_for_response,
            min_history_turns=int(self.configs.get("MIN_HISTORY_TURNS", 1)),
            max_history_turns=int(self.configs.get("MAX_HISTORY_TURNS", 8)),
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
        if not self.configs.get("CHECK_MODEL_CONTEXT", False):
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

        if not self.configs.get("AUTO_ADJUST_MODEL_CONTEXT", False):
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
        model_type = self.configs.get("MODEL_TYPE", "").lower()
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
        target_vram_usage = float(self.configs.get("TARGET_VRAM_USAGE", 0.8))
        n_gpu_layers = get_n_gpu_layers(
            model_path=model_path,
            configured_layers=self.configs.get("LAYERS", "auto"),
            n_ctx=configured_n_ctx or 2048,
            target_vram_usage=target_vram_usage,
        )

        # Get KV cache quantization setting (q8_0, q4_0, or f16 for full precision)
        kv_quant = self.configs.get("KV_CACHE_QUANT", "f16")
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

    def _get_key_matches(self, query: str, collection_name: str) -> list[dict[str, str]]:
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
        for chunk in chunks:
            normalized = chunk.strip()
            if not normalized:
                continue
            if normalized in seen:
                continue
            if self._is_low_quality_context_chunk(normalized):
                continue
            seen.add(normalized)
            filtered.append(normalized)
        return filtered

    def _search_collection(
        self,
        collection_name: str,
        query: str,
        filters: list[dict[str, object] | None],
        k: int | None = None,
    ) -> list[str]:
        if not query:
            return []
        if k is None:
            k = self.rag_k
        db = self._get_vector_db(collection_name)
        use_mmr = bool(self.configs.get("USE_MMR", True))
        fetch_k = int(self.configs.get("RAG_FETCH_K", 20))
        lambda_mult = float(self.configs.get("LAMBDA_MULT", 0.75))
        score_threshold = self.configs.get("RAG_SCORE_THRESHOLD")
        for where in filters:
            if use_mmr:
                if where is None:
                    docs_list = db.max_marginal_relevance_search(
                        query=query,
                        k=k,
                        # fetch_k must be >= k; if RAG_FETCH_K is smaller than k, use k as the floor
                        fetch_k=max(k, fetch_k),
                        lambda_mult=lambda_mult,
                    )
                else:
                    docs_list = db.max_marginal_relevance_search(
                        query=query,
                        k=k,
                        # fetch_k must be >= k; if RAG_FETCH_K is smaller than k, use k as the floor
                        fetch_k=max(k, fetch_k),
                        lambda_mult=lambda_mult,
                        filter=where,
                    )
                if docs_list or where is None:
                    return [doc.page_content for doc in docs_list]
            else:
                if where is None:
                    docs = db.similarity_search_with_score(query=query, k=k)
                else:
                    docs = db.similarity_search_with_score(query=query, k=k, filter=where)
                if score_threshold is not None:
                    docs = [(doc, score) for doc, score in docs if score <= float(score_threshold)]
                if docs or where is None:
                    return [doc.page_content for doc, _score in docs]
        return []

    def _get_vector_context(self, query: str, k: int | None = None) -> tuple[str, str]:
        if not self.rag_collection:
            return "", ""
        # Enrich query with character name to orient embedding search toward the character domain
        enriched_query = f"{self.character_name} {query}" if self.character_name else query
        matches = self._get_key_matches(query, self.rag_collection)
        filters = build_where_filters(matches)
        context_chunks = self._search_collection(self.rag_collection, enriched_query, filters, k=k)
        # Use unfiltered search for message examples: goal is stylistic match, not factual (Section 6.2)
        k_mes = k if k is not None else self.rag_k_mes
        mes_chunks = self._search_collection(f"{self.rag_collection}_mes", enriched_query, [None], k=k_mes)
        context_chunks = self._filter_context_chunks(context_chunks)
        mes_chunks = self._filter_context_chunks(mes_chunks)
        vector_context = "\n\n".join(context_chunks)
        mes_example = "\n\n".join(mes_chunks)
        return vector_context, mes_example

    def get_history(self) -> str:
        """Build conversation history from message deques."""
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
        return message_history

    def _build_mistral_prompt(self, message: str, vector_context: str, mes_example: str) -> str:
        # Build base system prompt WITHOUT vector context (it goes in final block only)
        voice_section = f"\n{self.voice_instructions}" if self.voice_instructions else ""
        system_prompt = (
            f"You are roleplaying as {self.character_name} in a continuous fictional chat with User. "
            "Stay in character, follow the description and scenario, and use the examples "
            f"and context as guidance.{voice_section}\n\n"
            f"{self.description}\n\n"
            f"Scenario:\n{self.scenario}\n\n"
            f"Message Examples:\n{mes_example}\n\n"
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
        if vector_context and vector_context.strip() != " ":
            current_content = f"{vector_context}\n\n{current_content}"
        blocks.append(f"<s>[INST] {current_content} [/INST]")
        return "\n".join(blocks)

    def _build_system_prompt_text(self, mes_example: str) -> str:
        """Build the system prompt text without vector context."""
        voice_section = f"\n{self.voice_instructions}" if self.voice_instructions else ""
        return (
            f"You are roleplaying as {self.character_name} in a continuous fictional chat with User. "
            "Stay in character, follow the description and scenario, and use the examples "
            f"and context as guidance.{voice_section}\n\n"
            f"{self.description}\n\n"
            f"Scenario:\n{self.scenario}\n\n"
            f"Message Examples:\n{mes_example}\n\n"
            "Do not repeat previous responses verbatim. Do not narrate static scene descriptions unless asked.\n\n"
            f"Reply only as {self.character_name}; never write any User lines "
            "(for example, never include 'User:' in your output) or dialogue for User."
        ).strip()

    def _prepare_vector_context(self, message: str) -> tuple[str, str]:
        """Prepare vector context with optional dynamic allocation."""
        vector_context = ""
        is_first_turn = not self.user_message_history and not self.ai_message_history
        mes_example = self.mes_example if is_first_turn else ""

        try:
            if self.use_dynamic_context and not is_first_turn:
                # Only apply dynamic allocation after first turn to avoid initial prompt bloat
                # Quick budget estimate to determine initial retrieval size
                system_prompt = self._build_system_prompt_text(self.mes_example if is_first_turn else "")
                budget = self.context_manager.calculate_budget(system_prompt)

                # Safety check: if budget for dynamic content is too small, fall back to static
                if budget.budget_for_dynamic_content < 500:
                    logger.warning(
                        "Available budget ({} tokens) too small for dynamic allocation. Using static mode.",
                        budget.budget_for_dynamic_content,
                    )
                    vector_context, mes_from_rag = self._get_vector_context(message)
                    if mes_from_rag:
                        mes_example = f"{mes_example}\n\n{mes_from_rag}".strip() if mes_example else mes_from_rag
                    elif not mes_example:
                        mes_example = self.mes_example
                else:
                    # Start with conservative k, expand if budget allows
                    chunk_size_estimate = int(self.configs.get("CHUNK_SIZE_ESTIMATE", 150))
                    context_budget_estimate = budget.budget_for_dynamic_content * 0.45  # 45% for context
                    initial_k = max(self.rag_k, int(context_budget_estimate / chunk_size_estimate))
                    max_initial = int(self.configs.get("MAX_INITIAL_RETRIEVAL", 20))
                    initial_k = min(initial_k, max_initial)  # Cap to avoid excessive queries

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
                    if self.configs.get("DEBUG_CONTEXT", False):
                        logger.debug(self.context_manager.get_context_info(budget, allocation))
            else:
                vector_context, mes_from_rag = self._get_vector_context(message)
                if mes_from_rag:
                    mes_example = f"{mes_example}\n\n{mes_from_rag}".strip() if mes_example else mes_from_rag
                elif not mes_example:
                    mes_example = self.mes_example
        except Exception as e:
            logger.warning("Error in dynamic context allocation: {}. Using static fallback.", e)
            vector_context = ""
            if not mes_example:
                mes_example = self.mes_example

        vector_context = (
            "[The following background information is relevant to the current topic. "
            "Use it to inform your response but do not quote it directly.]\n"
            f"{vector_context}"
        ) if vector_context else " "
        return vector_context, mes_example

    def _build_conversation_chain(self, message: str, vector_context: str, mes_example: str) -> tuple[object, object]:
        model_type = self.configs.get("MODEL_TYPE", "").lower()
        output_parser = StrOutputParser()
        if model_type == "mistral":
            prompt_text = self._build_mistral_prompt(message, vector_context, mes_example)
            if prompt_text.startswith("<s>"):
                prompt_text = prompt_text.removeprefix("<s>")
            if self.configs.get("DEBUG_PROMPT", False):
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
            if self.configs.get("DEBUG_PROMPT", False):
                logger.debug("Prompt template input: {}", query_input)
            conversation_chain = self.prompt | self.llm_model | output_parser
            chain_input = query_input
        return conversation_chain, chain_input

    async def _stream_response(  # noqa: PLR0912, PLR0915
        self,
        conversation_chain: object,
        chain_input: object,
        first_token_event: threading.Event | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> str | None:
        chunks: list[str] = []
        raw_stream = ""
        max_stream_chars = int(self.configs.get("MAX_STREAM_CHARS", 6000))
        max_silent_stream_chars = int(self.configs.get("MAX_SILENT_STREAM_CHARS", 200))
        empty_stream_fallback = str(
            self.configs.get(
                "EMPTY_STREAM_FALLBACK",
                "I am unable to produce a visible response right now. Please try again.",
            ),
        )
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
        self.user_message_history.append(message)
        self.ai_message_history.append(result)

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
            fallback_history_response = str(
                self.configs.get(
                    "QUALITY_FALLBACK_RESPONSE",
                    "I will not repeat myself. Ask your question with more specificity.",
                ),
            )
            self.update_history(message, fallback_history_response)
