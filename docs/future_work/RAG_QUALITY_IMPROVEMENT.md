# RAG Retrieval Quality — Current State and Next Work

Last verified: 2026-03-01

This document covers retrieval behavior in runtime inference (`core/conversation_manager.py`).

## Implemented

- Query enrichment: character name is prepended to the retrieval query when available.
- Metadata-aware filtering with staged fallbacks (`$and`, `$or`, then unfiltered).
- Alias-aware key matching in metadata filter construction.
- Separate retrieval counts for lore vs message examples (`RAG_K` and `RAG_K_MES`).
- Message-example retrieval intentionally runs unfiltered to prioritize stylistic signal.
- Optional MMR retrieval (`USE_MMR`) with configurable `RAG_FETCH_K` and `LAMBDA_MULT`.
- Optional score-threshold filtering when similarity mode is used (`RAG_SCORE_THRESHOLD`, `USE_MMR=false`).
- Context chunk cleanup pipeline:
  - low-quality pattern filtering,
  - near-duplicate filtering by signature,
  - section dedupe,
  - hard cap by `MAX_VECTOR_CONTEXT_CHARS`.
- Dynamic context budgeting is enabled by default and uses `ContextManager`.

## Remaining Gaps

- No cross-encoder reranker for second-stage ranking.
- No multi-query expansion/reformulation at retrieval time.
- No sentence-level compression pass before prompt injection.
- No cross-collection deduplication between main and `_mes` retrieval outputs.

## Current Retrieval-Relevant Defaults

From `configs/appconf.json`:

```json
{
  "RAG_COLLECTION": "shodan",
  "RAG_K": 3,
  "RAG_K_MES": 2,
  "RAG_FETCH_K": 8,
  "RAG_SCORE_THRESHOLD": 1.5,
  "USE_MMR": true,
  "LAMBDA_MULT": 0.75,
  "USE_DYNAMIC_CONTEXT": true,
  "MAX_INITIAL_RETRIEVAL": 8,
  "MAX_VECTOR_CONTEXT_CHARS": 2200
}
```

## Suggested Next Steps

1. Add optional reranking stage for top-N retrieved chunks.
2. Add retrieval telemetry (per-turn chunk count, score spread, filter path used).
3. Add evaluation fixtures for known queries and expected retrieval snippets.

## Related Files

- `core/conversation_manager.py`
- `core/context_manager.py`
- `scripts/rag/manage_collections.py`
- `tests/test_search_collection.py`
