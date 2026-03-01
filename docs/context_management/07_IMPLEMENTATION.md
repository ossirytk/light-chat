# Context Management Implementation Notes

Last verified: 2026-03-01

## Runtime Entry Points

- CLI chat: `main.py`
- Textual TUI chat: `chat_tui.py`

Both instantiate `ConversationManager` and use the same backend logic.

## Core Components

### `ContextManager` (`core/context_manager.py`)

- Computes available context budget.
- Allocates token budget across history/examples/context.
- Uses approximate token counting by default.

### `ConversationManager` (`core/conversation_manager.py`)

Responsibilities:

- Load configs and prompt/card data.
- Instantiate model with GPU layer logic + KV cache settings.
- Retrieve and clean RAG context.
- Build prompt per model type.
- Stream with guardrails.
- Post-process response and update history with quality gating.

## Important Behaviors

- Dynamic context is enabled by default.
- First turn uses character card examples (`mes_example`) directly.
- Later turns can inject retrieved `_mes` content.
- Prompt fingerprint logging is available for reproducibility checks.

## Test Coverage

- `tests/test_response_processing.py` validates streaming and post-processing quality gates.
- `tests/test_rag_scripts.py` validates metadata and ingestion helper behavior.

## Suggested Future Engineering

- Add deterministic retrieval regression fixtures.
- Add long-conversation stress tests.
- Add optional summarization module for old-history compression.
