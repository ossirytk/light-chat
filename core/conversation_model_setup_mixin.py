import contextlib
import json
from pathlib import Path

from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.prompts import BasePromptTemplate, load_prompt
from loguru import logger

from core.gpu_utils import get_n_gpu_layers


class UnknownModelTypeError(Exception):
    """Raised when an unsupported model type is configured."""


class ModelLoadError(Exception):
    """Raised when a model fails to load."""


class ConversationModelSetupMixin:
    def _replace_template_variables(self, text: str, user_name: str = "User", char_name: str = "") -> str:
        """Replace common template variables in character card text."""
        text = text.replace("{{user}}", user_name)
        text = text.replace("{{User}}", user_name)
        text = text.replace("{{char}}", char_name)
        return text.replace("{{Char}}", char_name)

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
