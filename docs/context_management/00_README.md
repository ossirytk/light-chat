# Context Management Docs

Last verified: 2026-03-12

This section documents dynamic context allocation used by runtime chat generation.

## Current Default Behavior

- Dynamic context is enabled by default: `USE_DYNAMIC_CONTEXT: true`.
- It is applied after the first turn.
- Older history is compacted into configurable summary entries (with topic-shift notes) once summarization thresholds are met.
- Small-talk and short follow-up turns may skip RAG retrieval.
- Retrieval can fan out across multiple deterministic query variants (`rag.multi_query.*`).
- Retrieval output is filtered, deduplicated, and capped before prompt injection.
- Retrieved lore can be compressed to the most query-relevant sentences (`context.retrieval.sentence_compression.*`).
- Optional reranking can reorder top retrieved chunks (`rag.rerank.*`).
- Optional retrieval telemetry can log per-turn retrieval stats (`rag.telemetry.enabled`).
- Optional persona drift scoring tracks long-session response fidelity (`conversation_quality.persona_drift.*`).

## Core Modules

- `core/context_manager.py` — token budgeting and allocation.
- `core/conversation_manager.py` — thin runtime orchestrator.
- `core/conversation_model_setup_mixin.py` — model/card/template loading and LlamaCpp setup.
- `core/conversation_prompt_history_mixin.py` — prompt assembly and history compaction/summaries.
- `core/conversation_response_mixin.py` — streaming guards, post-processing, quality checks, state I/O.
- `core/conversation_retrieval_mixin.py` — retrieval facade composed from backend/postprocess/orchestration mixins.
  - `core/conversation_retrieval_backend_mixin.py` — vector search, MMR/similarity, reranking.
  - `core/conversation_retrieval_keyfile_mixin.py` — metadata key loading and key-match extraction.
  - `core/conversation_retrieval_postprocess_mixin.py` — cleanup, dedupe, clipping, sentence compression.
  - `core/conversation_retrieval_orchestration_mixin.py` — skip logic, multi-query flow, telemetry, collection orchestration.

## Related Configuration Docs

- `docs/configs/00_README.md`

## Reading Order

1. `01_QUICKSTART.md`
2. `02_HOW_IT_WORKS.md`
3. `03_CONFIGURATION.md`
4. `04_EXAMPLES.md`
5. `05_VISUALIZATION.md`
6. `06_TROUBLESHOOTING.md`
7. `07_IMPLEMENTATION.md`
