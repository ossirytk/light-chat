import asyncio
import contextlib
import json
import logging
import threading
from collections import deque
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate, load_prompt
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from collection_helper import build_where_filters, extract_key_matches, normalize_keyfile
from context_manager import ApproximateTokenCounter, ContextManager


class UnknownModelTypeError(Exception):
    """Raised when an unsupported model type is configured."""


class ModelLoadError(Exception):
    """Raised when a model fails to load."""


class ConversationManager:
    def __init__(self) -> None:
        # Character card details
        self.character_name: str = ""
        self.description: str = ""
        self.scenario: str = ""
        self.mes_example: str = ""
        self.first_message: str = ""
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
        self.user_message_history: deque[str] = deque(maxlen=3)
        self.ai_message_history: deque[str] = deque(maxlen=3)
        self.persist_directory = self.configs.get("PERSIST_DIRECTORY", "./character_storage/")
        self.key_storage = self.configs.get("KEY_STORAGE", "./rag_data/")
        self.embedding_cache = self.configs.get("EMBEDDING_CACHE", "./embedding_models/")
        self.rag_k = int(self.configs.get("RAG_K", 7))
        self.rag_collection = self.configs.get("RAG_COLLECTION", self._sanitize_collection_name(self.character_name))
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

        description = self._replace_template_variables(description, char_name=char_name)
        scenario = self._replace_template_variables(scenario, char_name=char_name)
        mes_example = self._replace_template_variables(mes_example, char_name=char_name)
        first_message = self._replace_template_variables(first_message, char_name=char_name)

        self.description = description
        self.scenario = scenario
        self.mes_example = mes_example
        self.first_message = first_message

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
            stop_sequences = ["\nUser:", "User:", "\nUSER:", "USER:"]
            if self.character_name:
                stop_sequences.append(f"\n{self.character_name}:")
            add_bos = True

        model_kwargs = {"add_bos": add_bos}
        configured_n_ctx = self.configs.get("N_CTX")
        if configured_n_ctx is not None:
            configured_n_ctx = int(configured_n_ctx)

        llm_kwargs = {
            "model_path": model,
            "streaming": True,
            "last_n_tokens_size": int(self.configs["LAST_N_TOKENS_SIZE"]),
            "n_batch": int(self.configs["N_BATCH"]),
            "max_tokens": int(self.configs["MAX_TOKENS"]),
            "use_mmap": False,
            "top_p": float(self.configs["TOP_P"]),
            "top_k": int(self.configs["TOP_K"]),
            "temperature": float(self.configs["TEMPERATURE"]),
            "repeat_penalty": float(self.configs["REPEAT_PENALTY"]),
            "n_gpu_layers": int(self.configs["LAYERS"]),
            "rope_freq_scale": 1,
            "model_kwargs": model_kwargs,
            "stop": stop_sequences,
            "verbose": False,
        }
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
            encode_kwargs = {"normalize_embeddings": False}
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

    def _search_collection(self, collection_name: str, query: str, filters: list[dict[str, object]], k: int | None = None) -> list[str]:
        if not query:
            return []
        if k is None:
            k = self.rag_k
        db = self._get_vector_db(collection_name)
        for where in filters:
            docs = db.similarity_search_with_score(query=query, k=k, filter=where)
            if docs or where == {}:
                return [doc.page_content for doc, _score in docs]
        return []

    def _get_vector_context(self, query: str, k: int | None = None) -> tuple[str, str]:
        if not self.rag_collection:
            return "", ""
        matches = self._get_key_matches(query, self.rag_collection)
        filters = build_where_filters(matches)
        context_chunks = self._search_collection(self.rag_collection, query, filters, k=k)
        mes_chunks = self._search_collection(f"{self.rag_collection}_mes", query, filters, k=k)
        vector_context = "\n\n".join(chunk.strip() for chunk in context_chunks if chunk.strip())
        mes_example = "\n\n".join(chunk.strip() for chunk in mes_chunks if chunk.strip())
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
        system_prompt = (
            f"You are roleplaying as {self.character_name} in a continuous fictional chat with User. "
            "Stay in character, follow the description and scenario, and use the examples "
            f"and context as guidance. Reply only as {self.character_name}; never write any User lines "
            "(for example, never include 'User:' in your output) or dialogue for User.\n\n"
            f"{self.description}\n\n"
            f"Scenario:\n{self.scenario}\n\n"
            f"Message Examples:\n{mes_example}"
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
        return (
            f"You are roleplaying as {self.character_name} in a continuous fictional chat with User. "
            "Stay in character, follow the description and scenario, and use the examples "
            f"and context as guidance. Reply only as {self.character_name}; never write any User lines "
            "(for example, never include 'User:' in your output) or dialogue for User.\n\n"
            f"{self.description}\n\n"
            f"Scenario:\n{self.scenario}\n\n"
            f"Message Examples:\n{mes_example}"
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
                    
                    vector_context_full, mes_from_rag_full = self._get_vector_context(message, k=initial_k)
                    
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

        vector_context = f"Relevant context:\n{vector_context}" if vector_context else " "
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

    async def _stream_response(
        self,
        conversation_chain: object,
        chain_input: object,
        first_token_event: threading.Event | None = None,
    ) -> str | None:
        chunks = []
        char_name = self.character_name
        char_prefix = char_name + ": "
        prefix_len = len(char_name)
        prefix_done = False
        try:
            async for chunk in conversation_chain.astream(
                chain_input,
            ):
                if first_token_event is not None and not first_token_event.is_set():
                    first_token_event.set()
                chunks.append(chunk)
                if prefix_done:
                    print(chunk, flush=True, end="")  # noqa: T201
                elif len(chunks) >= prefix_len:
                    prefix_done = True
                    chunks_string = "".join(str(x) for x in chunks).strip()
                    if char_name in chunks_string:
                        print(chunks_string, flush=True, end="")  # noqa: T201
                    else:
                        print(char_prefix, flush=False, end="")  # noqa: T201
                        print(chunks_string, flush=True, end="")  # noqa: T201
        except (KeyboardInterrupt, asyncio.CancelledError):
            if first_token_event is not None:
                first_token_event.set()
            print()  # noqa: T201
            return None
        except Exception:
            # Suppress any underlying exceptions from llama_cpp during cleanup
            if first_token_event is not None:
                first_token_event.set()
            print()  # noqa: T201
            return None

        return "".join(str(x) for x in chunks)

    def update_history(self, message: str, result: str) -> None:
        """Update message history with user query and AI response."""
        self.user_message_history.append(message)
        self.ai_message_history.append(result)

    async def ask_question(self, message: str, first_token_event: threading.Event | None = None) -> None:
        """Query the model with streaming output."""
        if self.first_message and not self._greeting_in_history:
            self.user_message_history.append("")
            self.ai_message_history.append(self.first_message)
            self._greeting_in_history = True
        vector_context, mes_example = self._prepare_vector_context(message)
        conversation_chain, chain_input = self._build_conversation_chain(message, vector_context, mes_example)
        answer = await self._stream_response(conversation_chain, chain_input, first_token_event)
        if first_token_event is not None:
            first_token_event.set()
        if answer is None:
            return
        self.update_history(message, answer)
