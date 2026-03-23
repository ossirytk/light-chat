# Context Management Implementation Notes

Last verified: 2026-03-12

## Runtime Entry Points

- CLI chat: `main.py`
- Web chat: `web_app.py`

Both instantiate `ConversationManager` and use the same backend logic.

## Core Components

### `ContextManager` (`core/context_manager.py`)

- Computes available context budget.
- Allocates token budget across history/examples/context.
- Uses approximate token counting by default.

### `ConversationManager` Composition

Responsibilities:

- `core/conversation_manager.py`: runtime orchestrator and shared state setup.
- `core/conversation_model_setup_mixin.py`: config/card/template parsing and model instantiation.
- `core/conversation_retrieval_mixin.py`: retrieval facade.
	- `core/conversation_retrieval_backend_mixin.py`: vector DB/search/rerank backend.
	- `core/conversation_retrieval_postprocess_mixin.py`: retrieval filtering/dedupe/compression.
	- `core/conversation_retrieval_orchestration_mixin.py`: retrieval flow, telemetry, query expansion.
	- `core/conversation_retrieval_keyfile_mixin.py`: keyfile loading and alias/metadata matching.
- `core/conversation_prompt_history_mixin.py`: history summarization and prompt assembly.
- `core/conversation_response_mixin.py`: streaming guards, post-processing, quality gating, history/state API.

## Important Behaviors

- Dynamic context is enabled by default.
- First turn uses character card examples (`mes_example`) directly.
- Later turns can inject retrieved `_mes` content.
- Retrieval can expand into multiple deterministic query variants before results are merged.
- Retrieved lore can be sentence-compressed after cleanup and before prompt injection.
- Persona drift scoring is initialized in `ConversationManager` and recorded in `ConversationResponseMixin`.
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
- Calibrate persona-drift thresholds and reporting against recorded sessions.
