# Copilot Compact Reference — Implemented State

Last verified: 2026-03-26

Use this as the single compact reference for implemented work across conversation quality, RAG quality, and web app behavior.

## Conversation Quality (Implemented)

- Prompt guidance enforces in-character responses and disallows model-generated `User:` turns.
- `voice_instructions` from character cards is supported in prompt assembly.
- Dynamic context budgeting now propagates allocated history/context/examples into final prompt assembly.
- Streaming safeguards are active:
  - early stop on generated user-turn markers,
  - output cap via `MAX_STREAM_CHARS`,
  - silent stream guard via `MAX_SILENT_STREAM_CHARS`,
  - visible empty-stream fallback via `EMPTY_STREAM_FALLBACK`.
- Post-processing removes model-format artifacts and trims user-turn spillover.
- Response-quality gating runs before history write:
  - reject too-short responses,
  - reject responses containing user-turn markers,
  - reject exact duplicate of the previous AI response,
  - keep streamed output/history consistent on quality-gate fallback paths.
- Configurable history depth (`MAX_HISTORY_TURNS`, currently `10`).
- Conversation-turn segmentation in `ContextManager` is stabilized for reliable history budgeting.
- Long-history summarization is implemented with deterministic summary entries and configurable thresholds.
- Topic-shift annotations are included for lexically divergent adjacent summarized turns.
- Runtime persona drift scoring is implemented with hybrid heuristic + semantic-style trigram metrics.
- Conversation-state export/import includes persona drift history, latest score payload, rolling average, and per-turn drift trace data.
- Offline conversation quality evaluation is implemented via `uv run python -m scripts.conversation.evaluate_quality evaluate-conversation-fixtures`.
- Evaluator supports deterministic `mock` mode, optional `live` mode, JSON/CSV/history outputs, and baseline comparison with soft-fail thresholds.
- Session-based calibration tooling is implemented via `uv run python -m scripts.conversation.evaluate_quality calibrate-persona-drift`.
- Three conversation fixture packs are maintained:
  - `tests/fixtures/conversation_fixtures.json` — baseline Shodan cases
  - `tests/fixtures/conversation_fixtures_hard.json` — drift, style-break, and user-turn leakage cases for Shodan and Leonardo
  - `tests/fixtures/conversation_fixtures_negative.json` — template variable leakage, user-turn leakage, OOC forbidden-phrase assertions
- Canonical mock-mode baseline artifacts are captured via `uv run python -m scripts.conversation.capture_baselines capture-conversation-baselines` and stored under `logs/conversation_quality/baselines/`.
- Dynamic context is enabled by default.

Primary files:

- `core/conversation_manager.py`
- `core/persona_drift.py`
- `core/conversation_model_setup_mixin.py`
- `core/conversation_prompt_history_mixin.py`
- `core/conversation_response_mixin.py`
- `core/context_manager.py`
- `scripts/conversation/evaluate_quality.py`
- `scripts/conversation/capture_baselines.py`
- `tests/fixtures/conversation_fixtures.json`
- `tests/fixtures/conversation_fixtures_hard.json`
- `tests/fixtures/conversation_fixtures_negative.json`
- `tests/test_persona_drift.py`
- `tests/test_conversation_quality_eval.py`
- `tests/test_response_processing.py`
- `tests/test_history_summarization.py`

## RAG Data Quality (Implemented)

- `rag_data` sources support leading HTML comment headers for metadata.
- Ingestion strips leading metadata comments before embedding (`strip_leading_html_comment`).
- Metadata supports `category` and `aliases` in addition to UUID text mapping.
- Alias-aware metadata matching is implemented (`extract_key_matches`).
- Validation includes structural checks and duplicate UUID detection.
- Source-document coverage scoring is implemented and push blocks low-coverage metadata by default.
- Message-example linting is implemented for `*_message_examples.txt` formatting consistency.
- Category confidence thresholds are configurable during analysis/generation workflows.
- Embedding model benchmarking is implemented for retrieval-fixture comparison.
- Re-embedding migration tooling is implemented with atomic temp-collection switchover workflow.
- Collection-level embedding fingerprint metadata is stamped and mixed-model safety checks are enforced.
- Legacy fingerprint backfill tooling exists for older collections.
- Analysis tooling supports review and quality options:
  - `--auto-categories/--no-auto-categories`,
  - `--auto-aliases/--no-auto-aliases`,
  - `--max-aliases`,
  - `--strict`,
  - `--review-report`.

Primary files:

- `scripts/rag/analyze_rag_text.py`
- `scripts/rag/analyze_rag_coverage.py`
- `scripts/rag/lint_message_examples.py`
- `scripts/rag/benchmark_embedding_models.py`
- `scripts/rag/migrate_collection_embedding.py`
- `scripts/rag/push_rag_data.py`
- `scripts/rag/manage_collections.py`
- `tests/test_rag_scripts.py`
- `tests/test_rag_data_quality.py`
- `tests/test_manage_collections_backfill.py`

## Retrieval Quality (Implemented)

- Query enrichment prepends character name when available.
- Metadata-aware staged fallbacks are implemented (`$and`, `$or`, then unfiltered).
- Alias-aware matching is used in filter construction.
- Separate retrieval counts are used for lore vs message examples (`RAG_K`, `RAG_K_MES`).
- Message-example retrieval is intentionally unfiltered for style capture when used, and skipped in dynamic non-first-turn allocation paths to avoid redundant work.
- Optional MMR retrieval is available (`USE_MMR`, `RAG_FETCH_K`, `LAMBDA_MULT`).
- Optional score-threshold filtering is available when similarity mode is used (`RAG_SCORE_THRESHOLD`, `USE_MMR=false`).
- Retrieval cleanup pipeline includes low-quality filtering, near-duplicate filtering, section dedupe, and hard context cap (`MAX_VECTOR_CONTEXT_CHARS`).
- Dynamic context budgeting is enabled through `ContextManager`.
- Dynamic-context exception fallback performs static retrieval rather than dropping to empty context.
- Optional reranking stage is available (`rag.rerank.*`).
- Optional multi-query expansion/reformulation is available (`rag.multi_query.*`).
- Retrieval telemetry is available per turn (`rag.telemetry.enabled`).
- Optional sentence-level compression is available (`context.retrieval.sentence_compression.*`).
- Retrieval fixture coverage and opt-in live checks are implemented.
- Rerank runtime-win gate benchmark exists and is passing.
- Retrieval evaluation supports hard-gate thresholds via `--min-recall` and `--min-mrr` flags on `evaluate-fixtures`; exits non-zero when thresholds are not met.
- Retrieval trend rendering is available via `show-retrieval-trends` command — reads `logs/retrieval_eval/history.csv` and prints a compact table of Recall@k and MRR per run with deltas.
- Retrieval turn traces in web flow can include both retrieval metadata and latest persona-drift telemetry.

Primary files:

- `core/conversation_manager.py`
- `core/conversation_retrieval_mixin.py`
- `core/conversation_retrieval_backend_mixin.py`
- `core/conversation_retrieval_postprocess_mixin.py`
- `core/conversation_retrieval_orchestration_mixin.py`
- `core/conversation_retrieval_keyfile_mixin.py`
- `core/retrieval_keys.py`
- `core/context_manager.py`
- `scripts/rag/manage_collections.py`
- `tests/test_search_collection.py`
- `tests/test_retrieval_metrics.py`
- `tests/test_retrieval_fixtures.py`

## Web App (Implemented)

- FastAPI + Jinja2 + HTMX chat interface with streamed assistant output.
- Timeout handling and retry affordance in UI.
- Shared backend behavior through `ConversationManager`.
- Health diagnostics endpoints.
- Session save/load in web flow.
- Retrieval debug panel with collection/chunk/rerank details.
- Copy/export helpers (copy last, export TXT/JSON).
- UI equivalents for slash commands (`clear`, `reload`, `help`).
- Keyboard usability enhancements (prompt history and shortcuts).
- VS Code task/debug workflow for start/stop.
- In-UI session picker with naming support.
- Per-turn retrieval trace history in the debug panel.
- Session exports persist conversation-quality metadata and drift traces for later calibration.

Primary files:

- `web_app.py`
- `main.py`
- `templates/index.html`
- `templates/chat_message_pair.html`
- `templates/chat_messages.html`
- `templates/chat_single_message.html`

## Current Defaults Snapshot

From `configs/config.v2.json`:

```json
{
  "context": {
    "dynamic": {"enabled": true},
    "history": {
      "min_turns": 2,
      "max_turns": 10,
      "summarization": {
        "enabled": true,
        "threshold_turns": 8,
        "keep_recent_turns": 6,
        "max_entries": 12,
        "max_chars_per_turn": 140
      }
    },
    "retrieval": {
      "max_initial_retrieval": 8,
      "max_vector_context_chars": 2200,
      "sentence_compression": {
        "enabled": true,
        "max_sentences": 8
      }
    }
  },
  "generation": {
    "max_stream_chars": 2400,
    "max_silent_stream_chars": 120,
    "hard_max_tokens": 1024
  },
  "fallback": {
    "empty_stream": "I am unable to produce a visible response right now. Please try again.",
    "quality": "I will not repeat myself. Ask your question with more specificity."
  },
  "rag": {
    "collection": "shodan",
    "k": 3,
    "k_mes": 2,
    "fetch_k": 8,
    "score_threshold": 1.5,
    "use_mmr": true,
    "lambda_mult": 0.75,
    "rerank": {
      "enabled": true,
      "top_n": 8
    },
    "multi_query": {
      "enabled": true,
      "max_variants": 3
    }
  },
  "conversation_quality": {
    "persona_drift": {
      "enabled": true,
      "warning_threshold": 0.45,
      "fail_threshold": 0.65,
      "history_window": 20,
      "heuristic_weight": 0.6,
      "semantic_weight": 0.4
    }
  }
}
```

## Operational Commands

```bash
uv run ruff check .
uv run ruff format .
uv run python -m scripts.conversation.evaluate_quality evaluate-conversation-fixtures
uv run python -m scripts.conversation.evaluate_quality calibrate-persona-drift
uv run python -m scripts.conversation.capture_baselines capture-conversation-baselines
uv run python -m scripts.rag.manage_collections benchmark-rerank --require-runtime-win
uv run python -m scripts.rag.manage_collections show-retrieval-trends --history-csv logs/retrieval_eval/history.csv
uv run python -m scripts.quality_gate --skip-retrieval
uv run python -m scripts.quality_gate  # full gate (requires ChromaDB collections)
```

## Planning Source

Open work and forward-looking improvements are tracked in:

- `docs/future_work/REFINEMENTS.md`

## Unified Quality Gate (Implemented)

A single command runs all quality checks in sequence and prints a PASS/WARN/FAIL summary table.

```bash
# Skip retrieval (no ChromaDB required) - suitable for CI without collections
uv run python -m scripts.quality_gate --skip-retrieval

# Full gate with retrieval thresholds
uv run python -m scripts.quality_gate \
  --min-retrieval-recall 0.7 \
  --min-retrieval-mrr 0.5 \
  --strict
```

Steps in order:
1. **RAG data lint** — scans `rag_data/*_message_examples.txt` for style violations
2. **Retrieval fixtures** — general, hard, and negative packs with optional `--min-recall`/`--min-mrr` gate
3. **Conversation fixtures** — general, hard, and negative packs in deterministic mock mode; compares against baselines in `logs/conversation_quality/baselines/` when present

Flags:
- `--skip-retrieval` — skip retrieval steps (use in CI without live ChromaDB)
- `--strict` — promote WARN to FAIL
- `--min-retrieval-recall` / `--min-retrieval-mrr` — threshold gate on the general retrieval pack
- `--max-score-drop` / `--max-drift-increase` — regression thresholds vs conversation baselines (default: 0.08 each)
- `--retrieval-history-csv` / `--conversation-history-csv` — append metrics to history files
- `--baselines-dir` — override default `logs/conversation_quality/baselines/`

CI integration: `.github/workflows/quality_gate.yml` runs on every push/PR (no GPU required).

Primary files:
- `scripts/quality_gate.py`
- `scripts/conversation/capture_baselines.py`
- `.github/workflows/quality_gate.yml`
- `docs/quality_gate.md`
