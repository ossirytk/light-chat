# Context Management Implementation Notes

Last verified: 2026-03-07

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
- Compact older conversation turns into summary entries with topic-shift annotations.

## Important Behaviors

- Dynamic context is enabled by default.
- First turn uses character card examples (`mes_example`) directly.
- Later turns can inject retrieved `_mes` content.
- Prompt fingerprint logging is available for reproducibility checks.

## Test Coverage

- `tests/test_response_processing.py` validates streaming and post-processing quality gates.
- `tests/test_history_summarization.py` validates summary compaction, formatting, and topic-shift annotations.
- `tests/test_rag_scripts.py` validates metadata and ingestion helper behavior.
- `tests/test_search_collection.py` validates retrieval behavior (MMR/similarity/rerank/dedupe paths).
- `tests/test_retrieval_fixtures.py` validates fixture schema and opt-in live retrieval checks.
- `tests/test_retrieval_metrics.py` validates Recall@k/MRR and export helper functions.

## Suggested Future Engineering

- Add deterministic retrieval regression fixtures.
- Add long-conversation stress tests.
- Add persona-drift scoring and reporting for long-session QA.
