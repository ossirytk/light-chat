# config.v2.json

Last verified: 2026-03-06

File: `configs/config.v2.json`

## Purpose

Primary runtime configuration for model loading, generation, retrieval, context management, debug/logging, and script defaults.

## Top-Level Sections

- `character`
- `prompt`
- `paths`
- `embedding`
- `runtime`
- `model`
- `generation`
- `rag`
- `context`
- `heuristics`
- `conversation_quality`
- `fallback`
- `debug`
- `logging`

## Key Sections

### `character`

- `card`: character card filename under `cards/`

### `prompt`

- `template_file`: prompt template filename under `configs/`

### `paths`

- `persist_directory`: Chroma storage path
- `key_storage`: metadata key JSON directory
- `embedding_cache`: embedding model cache directory
- `documents_directory`: default source documents directory for scripts

### `embedding`

- `device`: embedding runtime device (for example `cpu`)
- `model`: sentence-transformers model id used for vector embedding (for example `sentence-transformers/all-mpnet-base-v2`)

### `model`

- `path`, `type`
- `layers` (`auto` or integer)
- `target_vram_usage`
- `kv_cache_quant` (`f16`, `q8_0`, `q4_0`)
- `n_ctx`, `n_batch`
- `context.check`, `context.auto_adjust`

### `generation`

- Sampling controls: `temperature`, `top_p`, `top_k`, `repeat_penalty`
- Output controls: `max_tokens`, `hard_max_tokens`
- Stream guards: `max_stream_chars`, `max_silent_stream_chars`

`max_tokens` is the configured generation target, while `hard_max_tokens` is an additional runtime safety cap exposed through `core/config.py`.

### `rag`

- Retrieval controls: `collection`, `k`, `k_mes`, `fetch_k`, `score_threshold`, `use_mmr`, `lambda_mult`
- Reranker controls: `rerank.enabled`, `rerank.model`, `rerank.top_n`
- Query expansion: `multi_query.enabled`, `multi_query.max_variants`
- Telemetry: `telemetry.enabled`
- Ingestion defaults: `chunk_size`, `chunk_overlap`

### `context`

- Dynamic mode: `dynamic.enabled`
- Budget: `budget.reserved_for_response`
- History: `history.min_turns`, `history.max_turns`
- Retrieval sizing: `retrieval.chunk_size_estimate`, `retrieval.max_initial_retrieval`, `retrieval.max_vector_context_chars`
- Retrieval compression: `retrieval.sentence_compression.enabled`, `retrieval.sentence_compression.max_sentences`

### `heuristics`

- `small_talk_max_words`
- `followup_rag_max_words`

### `conversation_quality`

- Persona drift scoring: `persona_drift.enabled`
- Thresholds: `persona_drift.warning_threshold`, `persona_drift.fail_threshold`
- Rolling history window: `persona_drift.history_window`
- Weighting: `persona_drift.heuristic_weight`, `persona_drift.semantic_weight`

### `fallback`

- `empty_stream`
- `quality`

### `debug`

- `prompt`
- `prompt_fingerprint`
- `context`

### `logging`

- `level`
- `show_logs`
- `to_file`
- `file`

## Notes

- `rag.score_threshold` applies in similarity mode (`use_mmr=false`).
- Rerank can increase first-pass retrieval pool via `rag.rerank.top_n`.
- Multi-query expansion can issue several deterministic query variants before merging results.
- Sentence compression can reduce injected vector context to the highest-overlap sentences.
- Persona drift settings feed long-session response scoring and telemetry in `ConversationManager`.
- `context.dynamic.enabled` affects primarily non-first turns.
- Invalid/missing values are type-coerced with safe defaults in `core/config.py`.
- The values documented here are the structure of `configs/config.v2.json`; runtime fallback defaults live in `core/config.py`.
