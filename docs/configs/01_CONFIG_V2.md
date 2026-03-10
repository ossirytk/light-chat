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

### `rag`

- Retrieval controls: `collection`, `k`, `k_mes`, `fetch_k`, `score_threshold`, `use_mmr`, `lambda_mult`
- Reranker controls: `rerank.enabled`, `rerank.model`, `rerank.top_n`
- Telemetry: `telemetry.enabled`
- Ingestion defaults: `chunk_size`, `chunk_overlap`

### `context`

- Dynamic mode: `dynamic.enabled`
- Budget: `budget.reserved_for_response`
- History: `history.min_turns`, `history.max_turns`
- Retrieval sizing: `retrieval.chunk_size_estimate`, `retrieval.max_initial_retrieval`, `retrieval.max_vector_context_chars`

### `heuristics`

- `small_talk_max_words`
- `followup_rag_max_words`

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
- `context.dynamic.enabled` affects primarily non-first turns.
- Invalid/missing values are type-coerced with safe defaults in `core/config.py`.
