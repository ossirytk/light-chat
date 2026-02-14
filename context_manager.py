"""Dynamic context window management for LLM prompt construction.

This module provides utilities to gauge available prompt context and dynamically
allocate tokens between system prompt, message examples, RAG context, and
conversation history based on the model's context window.
"""

import logging
from dataclasses import dataclass
from typing import Protocol

from loguru import logger


class TokenCounterProtocol(Protocol):
    """Protocol for token counting implementations."""

    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""


@dataclass
class ContextBudget:
    """Represents available token budget for different prompt components."""

    total_context: int
    reserved_for_response: int
    system_prompt_tokens: int
    available_for_context: int

    @property
    def max_user_input_tokens(self) -> int:
        """Approximate max tokens for user input (conservative estimate)."""
        return min(self.available_for_context // 10, 500)

    @property
    def budget_for_dynamic_content(self) -> int:
        """Budget remaining after system prompt and input."""
        dynamic_budget = self.available_for_context - self.max_user_input_tokens
        return max(0, dynamic_budget)


class ApproximateTokenCounter:
    """Approximate token counter using character-based heuristics.

    Useful when no tokenizer is available. Approximates ~1 token per 4 characters
    for English text, with adjustments for special tokens and formatting.
    """

    @staticmethod
    def count_tokens(text: str) -> int:
        """Estimate token count using character-based heuristics."""
        if not text:
            return 0

        # Basic character to token ratio (~1 token per 4 chars for English)
        char_based_estimate = len(text) // 4

        # Count special tokens (rough estimate)
        special_token_count = (
            text.count("\n") // 2 + text.count("<|") + text.count("[INST]") * 2 + text.count("<s>")
        )

        # Combine estimates
        return max(1, char_based_estimate + special_token_count)


class ExactTokenCounter:
    """Exact token counter using a HuggingFace tokenizer."""

    def __init__(self, tokenizer: object) -> None:
        """Initialize with a HuggingFace tokenizer.

        Args:
            tokenizer: A HuggingFace tokenizer with an encode method.
        """
        self.tokenizer = tokenizer

    def count_tokens(self, text: str) -> int:
        """Count tokens using the tokenizer."""
        if not text:
            return 0
        try:
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning("Failed to count tokens with exact tokenizer, falling back: {}", e)
            return ApproximateTokenCounter.count_tokens(text)


class ContextManager:
    """Manages dynamic context allocation for LLM prompts.

    Analyzes available context window and intelligently distributes tokens
    between system prompt, message examples, RAG context, and conversation history.
    """

    def __init__(
        self,
        context_window: int,
        token_counter: TokenCounterProtocol | None = None,
        reserved_for_response: int = 256,
        min_history_turns: int = 1,
        max_history_turns: int = 8,
    ) -> None:
        """Initialize the context manager.

        Args:
            context_window: Model's total context window size in tokens.
            token_counter: Token counter implementation. Defaults to approximate counter.
            reserved_for_response: Tokens reserved for model response. Default 256.
            min_history_turns: Minimum conversation turns to preserve. Default 1.
            max_history_turns: Maximum conversation turns to use. Default 8.
        """
        self.context_window = context_window
        self.token_counter = token_counter or ApproximateTokenCounter()
        self.reserved_for_response = reserved_for_response
        self.min_history_turns = min_history_turns
        self.max_history_turns = max_history_turns

        logging.getLogger("transformers").setLevel(logging.ERROR)

    def calculate_budget(self, system_prompt: str) -> ContextBudget:
        """Calculate available token budget for dynamic content.

        Args:
            system_prompt: The system/instruction prompt text.

        Returns:
            ContextBudget with allocated token counts.
        """
        system_tokens = self.token_counter.count_tokens(system_prompt)
        available_context = self.context_window - self.reserved_for_response - system_tokens

        if available_context < 0:
            logger.warning(
                "System prompt ({} tokens) + response buffer ({}) exceeds context window ({}). "
                "Consider reducing system prompt or increasing context window.",
                system_tokens,
                self.reserved_for_response,
                self.context_window,
            )
            available_context = 0

        return ContextBudget(
            total_context=self.context_window,
            reserved_for_response=self.reserved_for_response,
            system_prompt_tokens=system_tokens,
            available_for_context=available_context,
        )

    def allocate_content(
        self,
        budget: ContextBudget,
        message_examples: str,
        vector_context: str,
        conversation_history: str,
        current_input: str,
    ) -> dict[str, str | int]:
        """Dynamically allocate tokens among content components.

        Prioritizes in order:
        1. Current user input (required)
        2. Minimum conversation history
        3. Message examples
        4. Vector context (RAG)

        Args:
            budget: Available token budget.
            message_examples: Message/example text to include.
            vector_context: Vector search results context.
            conversation_history: Previous conversation turns.
            current_input: Current user message.

        Returns:
            Dict with allocated content and token counts.
        """
        input_tokens = self.token_counter.count_tokens(current_input)
        remaining_budget = budget.budget_for_dynamic_content - input_tokens

        if remaining_budget < 0:
            logger.warning(
                "Current input ({} tokens) exceeds dynamic budget ({}). Will be truncated.",
                input_tokens,
                budget.budget_for_dynamic_content,
            )
            return {
                "allocated_input": current_input,
                "allocated_examples": "",
                "allocated_context": "",
                "allocated_history": "",
                "input_tokens": input_tokens,
                "examples_tokens": 0,
                "context_tokens": 0,
                "history_tokens": 0,
                "total_allocated": input_tokens,
            }

        # Allocate conversation history (minimum priority to preserve coherence)
        history_allocation = self._allocate_history(
            conversation_history,
            remaining_budget * 0.3,  # Reserve 30% for history
        )
        remaining_budget -= history_allocation["tokens"]

        # Allocate message examples
        examples_allocation = self._allocate_content(
            message_examples,
            remaining_budget * 0.25,  # 25% for examples
        )
        remaining_budget -= examples_allocation["tokens"]

        # Allocate vector context (remaining budget)
        context_allocation = self._allocate_content(vector_context, remaining_budget)

        total_allocated = (
            input_tokens
            + history_allocation["tokens"]
            + examples_allocation["tokens"]
            + context_allocation["tokens"]
        )

        logger.debug(
            "Context allocation: input={} hist={} ex={} ctx={} (total={}/{})",
            input_tokens,
            history_allocation["tokens"],
            examples_allocation["tokens"],
            context_allocation["tokens"],
            total_allocated,
            budget.available_for_context,
        )

        return {
            "allocated_input": current_input,
            "allocated_examples": examples_allocation["content"],
            "allocated_context": context_allocation["content"],
            "allocated_history": history_allocation["content"],
            "input_tokens": input_tokens,
            "examples_tokens": examples_allocation["tokens"],
            "context_tokens": context_allocation["tokens"],
            "history_tokens": history_allocation["tokens"],
            "total_allocated": total_allocated,
        }

    def _allocate_content(self, content: str, max_tokens: int) -> dict[str, str | int]:
        """Allocate a portion of content to fit within token budget.

        Args:
            content: Content to allocate.
            max_tokens: Maximum tokens to use.

        Returns:
            Dict with allocated content and token count.
        """
        if max_tokens <= 0 or not content:
            return {"content": "", "tokens": 0}

        content_tokens = self.token_counter.count_tokens(content)

        if content_tokens <= max_tokens:
            return {"content": content, "tokens": content_tokens}

        # Truncate by character-based ratio to fit token budget
        char_ratio = len(content) / max(1, content_tokens)
        target_chars = int(max_tokens * char_ratio * 0.9)  # 90% to be conservative

        truncated = content[:target_chars].rsplit("\n", 1)[0]  # Truncate at paragraph boundary

        return {
            "content": truncated,
            "tokens": self.token_counter.count_tokens(truncated),
        }

    def _allocate_history(
        self,
        conversation_history: str,
        max_tokens: int,
    ) -> dict[str, str | int]:
        """Allocate conversation history respecting turn boundaries.

        Prioritizes recent turns and preserves at least min_history_turns.

        Args:
            conversation_history: Full conversation history.
            max_tokens: Maximum tokens to allocate.

        Returns:
            Dict with allocated history and token count.
        """
        if max_tokens <= 0 or not conversation_history:
            return {"content": "", "tokens": 0}

        # Split into turns (User: ... \nCharacter:...)
        turns = self._split_conversation_turns(conversation_history)

        if not turns:
            return {"content": "", "tokens": 0}

        # Start with recent turns and work backward
        included_turns: list[str] = []
        remaining_tokens = max_tokens

        for turn in reversed(turns):
            turn_tokens = self.token_counter.count_tokens(turn)

            if len(included_turns) < self.min_history_turns:
                # Always include minimum turns
                included_turns.insert(0, turn)
                remaining_tokens -= turn_tokens
            elif turn_tokens <= remaining_tokens and len(included_turns) < self.max_history_turns:
                included_turns.insert(0, turn)
                remaining_tokens -= turn_tokens
            else:
                break

        allocated_history = "".join(included_turns)

        return {
            "content": allocated_history,
            "tokens": self.token_counter.count_tokens(allocated_history),
        }

    @staticmethod
    def _split_conversation_turns(history: str) -> list[str]:
        """Split conversation history into individual turns.

        Args:
            history: Full conversation history string.

        Returns:
            List of conversation turns.
        """
        if not history:
            return []

        turns: list[str] = []
        current_turn = ""

        for line in history.split("\n"):
            current_turn += line + "\n"

            # A turn ends when we see the start of a new User: line
            if current_turn.strip().startswith("User:") and current_turn.count("\n") > 1:
                turns.append(current_turn.rsplit("\n", 1)[0] + "\n")
                current_turn = line + "\n"

        if current_turn.strip():
            turns.append(current_turn)

        return turns

    def get_context_info(self, budget: ContextBudget, allocation: dict) -> str:
        """Get a human-readable summary of context allocation.

        Args:
            budget: The context budget.
            allocation: The allocation result from allocate_content.

        Returns:
            Formatted string describing the context allocation.
        """
        return (
            f"Context Window: {budget.total_context} tokens\n"
            f"  Response Buffer: {budget.reserved_for_response} tokens\n"
            f"  System Prompt: {budget.system_prompt_tokens} tokens\n"
            f"  Available: {budget.available_for_context} tokens\n"
            f"Allocation:\n"
            f"  Input: {allocation['input_tokens']} tokens\n"
            f"  History: {allocation['history_tokens']} tokens\n"
            f"  Examples: {allocation['examples_tokens']} tokens\n"
            f"  Context: {allocation['context_tokens']} tokens\n"
            f"  Total: {allocation['total_allocated']} / {budget.available_for_context}"
        )
