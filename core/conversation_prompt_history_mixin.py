import re

from langchain_core.output_parsers import StrOutputParser
from loguru import logger


class ConversationPromptHistoryMixin:
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
        return f"- User asked about {topic_label}: {user_text} | {self.character_name}: {ai_text}{topic_shift_note}"

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

    def _build_mistral_prompt(
        self,
        message: str,
        vector_context: str,
        mes_example: str,
        history_override: str | None = None,
    ) -> str:
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

        if history_override is not None:
            history_turn_count = history_override.count("\nUser:")
            if history_override.startswith("User:"):
                history_turn_count += 1
            if history_turn_count <= 0:
                user_message_history_list = []
                ai_message_history_list = []
            else:
                user_message_history_list = user_message_history_list[-history_turn_count:]
                ai_message_history_list = ai_message_history_list[-history_turn_count:]

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

    def _prepare_dynamic_vector_context(self, message: str, mes_example: str) -> tuple[str, str, str]:
        system_prompt = self._build_system_prompt_text("")
        budget = self.context_manager.calculate_budget(system_prompt)

        if budget.budget_for_dynamic_content < self._MIN_DYNAMIC_CONTENT_TOKENS:
            logger.warning(
                "Available budget ({} tokens) too small for dynamic allocation. Using static mode.",
                budget.budget_for_dynamic_content,
            )
            vector_context, mes_example = self._prepare_static_vector_context(message, mes_example)
            return vector_context, mes_example, self.get_history()

        chunk_size_estimate = self.runtime_config.chunk_size_estimate
        context_budget_estimate = budget.budget_for_dynamic_content * 0.25
        initial_k = max(self.rag_k, int(context_budget_estimate / chunk_size_estimate))
        max_initial = self.runtime_config.max_initial_retrieval
        initial_k = min(initial_k, max_initial)

        vector_context_full, _ = self._get_vector_context(message, k=initial_k, include_mes=False)
        history = self.get_history()
        allocation = self.context_manager.allocate_content(
            budget,
            "",
            vector_context_full,
            history,
            message,
        )
        vector_context = str(allocation["allocated_context"])
        allocated_history = str(allocation["allocated_history"])

        if self.runtime_config.debug_context:
            logger.debug(self.context_manager.get_context_info(budget, allocation))

        return vector_context, mes_example, allocated_history

    def _prepare_vector_context(self, message: str) -> tuple[str, str, str | None]:
        """Prepare vector context with optional dynamic allocation."""
        vector_context = ""
        is_first_turn = not self.user_message_history and not self.ai_message_history
        mes_example = self.mes_example if is_first_turn else ""
        allocated_history: str | None = None

        if self._should_skip_rag_for_message(message):
            return " ", mes_example, allocated_history
        if (not is_first_turn) and self._should_skip_rag_for_followup(message):
            return " ", mes_example, allocated_history

        try:
            if self.use_dynamic_context and not is_first_turn:
                vector_context, mes_example, allocated_history = self._prepare_dynamic_vector_context(
                    message,
                    mes_example,
                )
            else:
                vector_context, mes_example = self._prepare_static_vector_context(message, mes_example)
        except Exception as e:
            logger.warning("Error in dynamic context allocation: {}. Using static fallback.", e)
            vector_context, mes_example = self._prepare_static_vector_context(message, mes_example)

        vector_context = (
            (
                "[The following background information is relevant to the current topic. "
                "Use it to inform your response but do not quote it directly.]\n"
                f"{vector_context}"
            )
            if vector_context
            else " "
        )
        return vector_context, mes_example, allocated_history

    def _build_conversation_chain(
        self,
        message: str,
        vector_context: str,
        mes_example: str,
        history_override: str | None = None,
    ) -> tuple[object, object]:
        model_type = self.runtime_config.model_type.lower()
        output_parser = StrOutputParser()
        if model_type == "mistral":
            prompt_text = self._build_mistral_prompt(message, vector_context, mes_example, history_override)
            if prompt_text.startswith("<s>"):
                prompt_text = prompt_text.removeprefix("<s>")
            if self.runtime_config.debug_prompt:
                logger.debug("Mistral prompt:\n{}", prompt_text)
            conversation_chain = self.llm_model | output_parser
            chain_input = prompt_text
        else:
            history = history_override if history_override is not None else self.get_history()
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
