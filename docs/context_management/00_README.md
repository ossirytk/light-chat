# Context Management Docs

Last verified: 2026-03-01

This section documents dynamic context allocation used by runtime chat generation.

## Current Default Behavior

- Dynamic context is enabled by default: `USE_DYNAMIC_CONTEXT: true`.
- It is applied after the first turn.
- Small-talk and short follow-up turns may skip RAG retrieval.
- Retrieval output is filtered, deduplicated, and capped before prompt injection.

## Core Modules

- `core/context_manager.py` — token budgeting and allocation.
- `core/conversation_manager.py` — retrieval + prompt assembly + streaming.

## Reading Order

1. `01_QUICKSTART.md`
2. `02_HOW_IT_WORKS.md`
3. `03_CONFIGURATION.md`
4. `04_EXAMPLES.md`
5. `05_VISUALIZATION.md`
6. `06_TROUBLESHOOTING.md`
7. `07_IMPLEMENTATION.md`
