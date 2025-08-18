import json
from collections import deque
from pathlib import Path

from icecream import ic
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate, load_prompt


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
        self.llama_input: str = ""
        self.llama_instruction: str = ""
        self.llama_response: str = ""
        self.llama_endtoken: str = ""

        # Init things
        self.configs = self.load_configs()
        self.prompt = self.parse_prompt()
        self.llm_model = self.instantiate_llm()
        self.user_message_history: deque[str] = deque(maxlen=3)
        self.ai_message_history: deque[str] = deque(maxlen=3)

    def load_configs(self) -> dict:
        config_source = Path("./configs/") / "modelconf.json"
        ic(f"Loading configs from: {config_source}")
        with config_source.open() as f:
            config_file = f.read()
            config_json = json.loads(config_file)
        ic(type(config_json))
        return config_json

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
                "instruction": "[INST]",
                "input": "",
                "response": "[/INST]",
                "endtoken": "",
            },
            "chatml": {
                "instruction": "<|system|>",
                "input": "<|user|>",
                "response": "<|assistant|>",
                "endtoken": "</s>",
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

    def _load_character_card(self) -> dict:
        """Load and parse the character card JSON file."""
        card_source = Path("./cards/") / self.configs["CHARACTER_CARD"]
        ic(f"Loading card from: {card_source}")
        with card_source.open() as f:
            card_file = f.read()
            card_json = json.loads(card_file)
        return card_json["data"]

    def _load_prompt_template(self) -> BasePromptTemplate:
        """Load the prompt template from JSON file."""
        prompt_template = Path("./configs/") / self.configs["PROMPT_TEMPLATE"]
        ic(f"Loading prompt from: {prompt_template}")
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

        # Extract and process card fields
        description = card["description"] if "description" in card else card["char_persona"]
        scenario = card["scenario"] if "scenario" in card else card["world_scenario"]
        mes_example = card["mes_example"] if "mes_example" in card else card["example_dialogue"]
        first_message = card["first_mes"] if "first_mes" in card else card["char_greeting"]

        description = self._replace_template_variables(description, char_name=char_name)
        scenario = self._replace_template_variables(scenario, char_name=char_name)
        mes_example = self._replace_template_variables(mes_example, char_name=char_name)
        first_message = self._replace_template_variables(first_message, char_name=char_name)

        self.description = description
        self.scenario = scenario
        self.mes_example = mes_example

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
        # TODO better selector for model
        model_path = Path("./models/") / self.configs["MODEL"]
        model = str(model_path)
        if not model:
            msg = "Could not load the model"
            raise ModelLoadError(msg)

        ic(f"loading model from: {model}")
        # Add things here if you want to play with the model params
        # MAX_TOKENS is an optional param for when model answer cuts off
        # This can happen when large context models are told to print multiple paragraphs
        # Setting MAX_TOKENS lower than the context size can sometimes fix this

        # TODO move the variables above for clarity and error handling
        return LlamaCpp(
            model_path=model,
            streaming=True,
            n_ctx=int(self.configs["N_CTX"]),
            last_n_tokens_size=int(self.configs["LAST_N_TOKENS_SIZE"]),
            n_batch=int(self.configs["N_BATCH"]),
            max_tokens=int(self.configs["MAX_TOKENS"]),
            use_mmap=False,
            top_p=float(self.configs["TOP_P"]),
            top_k=int(self.configs["TOP_K"]),
            temperature=float(self.configs["TEMPERATURE"]),
            repeat_penalty=float(self.configs["REPEAT_PENALTY"]),
            n_gpu_layers=int(self.configs["LAYERS"]),
            rope_freq_scale=1,
            verbose=False,
        )

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

    def update_history(self, message: str, result: str) -> None:
        """Update message history with user query and AI response."""
        self.user_message_history.append(message)
        self.ai_message_history.append(result)

    async def ask_question_test(self, message: str) -> None:
        """Query the model with streaming output."""
        history = self.get_history()
        query_input = {
            "input": message,
            "history": history,
        }

        output_parser = StrOutputParser()
        conversation_chain = self.prompt | self.llm_model | output_parser

        chunks = []
        char_name = self.character_name
        char_prefix = char_name + ": "
        prefix_len = len(char_name)
        prefix_done = False
        # TODO vector memory
        # TODO GLiNER parsing?
        # TODO consider better output formatting
        async for chunk in conversation_chain.astream(
            query_input,
        ):
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

        answer = "".join(str(x) for x in chunks)
        self.update_history(message, answer)
