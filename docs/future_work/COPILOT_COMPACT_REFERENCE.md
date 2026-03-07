# Copilot Compact Reference — Implemented State

Last verified: 2026-03-09

Use this as the single compact reference for implemented work across conversation quality, RAG quality, and web app behavior.

## Conversation Quality (Implemented)

- Prompt guidance enforces in-character responses and disallows model-generated `User:` turns.
- `voice_instructions` from character cards is supported in prompt assembly.
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
  - persist `QUALITY_FALLBACK_RESPONSE` when checks fail.
- Configurable history depth (`MAX_HISTORY_TURNS`, currently `10`).
- Long-history summarization is implemented with deterministic summary entries and configurable thresholds.
- Topic-shift annotations are included for lexically divergent adjacent summarized turns.
- Dynamic context is enabled by default.

Primary files:

- `core/conversation_manager.py`
- `core/context_manager.py`
- `tests/test_response_processing.py`
- `tests/test_history_summarization.py`

## RAG Data Quality (Implemented)

- `rag_data` sources support leading HTML comment headers for metadata.
- Ingestion strips leading metadata comments before embedding (`strip_leading_html_comment`).
- Metadata supports `category` and `aliases` in addition to UUID text mapping.
- Alias-aware metadata matching is implemented (`extract_key_matches`).
- Validation includes structural checks and duplicate UUID detection.
- Analysis tooling supports review and quality options:
  - `--auto-categories/--no-auto-categories`,
  - `--auto-aliases/--no-auto-aliases`,
  - `--max-aliases`,
  - `--strict`,
  - `--review-report`.

Primary files:

- `scripts/rag/analyze_rag_text.py`
- `scripts/rag/push_rag_data.py`
- `scripts/rag/manage_collections.py`
- `tests/test_rag_scripts.py`

## Retrieval Quality (Implemented)

- Query enrichment prepends character name when available.
- Metadata-aware staged fallbacks are implemented (`$and`, `$or`, then unfiltered).
- Alias-aware matching is used in filter construction.
- Separate retrieval counts are used for lore vs message examples (`RAG_K`, `RAG_K_MES`).
- Message-example retrieval is intentionally unfiltered for style capture.
- Optional MMR retrieval is available (`USE_MMR`, `RAG_FETCH_K`, `LAMBDA_MULT`).
- Optional score-threshold filtering is available when similarity mode is used (`RAG_SCORE_THRESHOLD`, `USE_MMR=false`).
- Retrieval cleanup pipeline includes low-quality filtering, near-duplicate filtering, section dedupe, and hard context cap (`MAX_VECTOR_CONTEXT_CHARS`).
- Dynamic context budgeting is enabled through `ContextManager`.
- Optional reranking stage is available (`rag.rerank.*`).
- Optional multi-query expansion/reformulation is available (`rag.multi_query.*`).
- Retrieval telemetry is available per turn (`rag.telemetry.enabled`).
- Optional sentence-level compression is available (`context.retrieval.sentence_compression.*`).
- Retrieval fixture coverage and opt-in live checks are implemented.
- Rerank runtime-win gate benchmark exists and is passing.

Primary files:

- `core/conversation_manager.py`
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
    "max_stream_chars": 800,
    "max_silent_stream_chars": 120
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
    "multi_query": {
      "enabled": true,
      "max_variants": 3
    }
  }
}
```

## Operational Commands

```bash
uv run ruff check .
uv run ruff format .
uv run python -m scripts.rag.manage_collections benchmark-rerank --require-runtime-win
```

## Planning Source

Open work and forward-looking improvements are tracked in:

- `docs/future_work/REFINEMENTS.md`
