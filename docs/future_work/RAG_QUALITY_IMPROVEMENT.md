# RAG Retrieval Quality — Current State and Next Work

Last verified: 2026-03-07

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
- Optional reranking stage is available for top-N candidates (`rag.rerank.*`).
- Optional multi-query expansion/reformulation is available at retrieval time (`rag.multi_query.*`).
- Retrieval telemetry is available per turn (`rag.telemetry.enabled`).
- Optional sentence-level compression is available before prompt injection (`context.retrieval.sentence_compression.*`).
- Retrieval evaluation fixtures are available in `tests/fixtures/retrieval_fixtures.json` with opt-in live checks in `tests/test_retrieval_fixtures.py`.

## Remaining Gaps

- No automated pass/fail thresholds yet for hard/general fixture packs (only rerank benchmark has an explicit gate).
- No persisted longitudinal dashboard visualization yet (CSV history exists, but no chart/report generator).

## Current Retrieval-Relevant Defaults

From `configs/config.v2.json`:

```json
{
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
  },
  "context": {
    "dynamic": {"enabled": true},
    "retrieval": {
      "max_initial_retrieval": 8,
      "max_vector_context_chars": 2200,
      "sentence_compression": {
        "enabled": true,
        "max_sentences": 8
      }
    }
  }
}
```

## Suggested Next Steps

1. Add optional pass/fail thresholds for hard and general fixture packs.
2. Add a lightweight trend renderer over `logs/retrieval_eval/history.csv`.

## Benchmark Snapshot (2026-03-07)

### Hard Fixture Pack (`tests/fixtures/retrieval_fixtures_hard.json`)

- Similarity mode: Recall@7 = 0.750, MRR = 0.688, ExpRecall@7 = 0.750, Precision@7 = 0.143, MAP@7 = 0.287
- Runtime mode: Recall@7 = 0.750, MRR = 0.604, ExpRecall@7 = 0.750, Precision@7 = 0.143, MAP@7 = 0.250

### Rerank Fixture Pack (`tests/fixtures/retrieval_fixtures_rerank.json`)

- Similarity mode: Recall@5 = 0.667, MRR = 0.194, ExpRecall@5 = 0.667, Precision@5 = 0.133, MAP@5 = 0.153
- Runtime mode: Recall@5 = 1.000, MRR = 1.000, ExpRecall@5 = 1.000, Precision@5 = 0.200, MAP@5 = 0.833
- Gate status (`benchmark-rerank --require-runtime-win`): PASS

## Related Files

- `core/conversation_manager.py`
- `core/context_manager.py`
- `scripts/rag/manage_collections.py`
- `tests/test_search_collection.py`
- `tests/test_retrieval_metrics.py`
