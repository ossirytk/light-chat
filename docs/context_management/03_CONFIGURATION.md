# Context Management Configuration

Last verified: 2026-03-01

This page lists context-related keys currently used in runtime code.

## Main Toggles

```json
{
  "USE_DYNAMIC_CONTEXT": true,
  "DEBUG_CONTEXT": false,
  "DEBUG_PROMPT": true,
  "DEBUG_PROMPT_FINGERPRINT": true
}
```

## Budget and History

```json
{
  "RESERVED_FOR_RESPONSE": 384,
  "MIN_HISTORY_TURNS": 2,
  "MAX_HISTORY_TURNS": 10,
  "MAX_STREAM_CHARS": 800,
  "MAX_SILENT_STREAM_CHARS": 120
}
```

## Retrieval Controls

```json
{
  "RAG_COLLECTION": "shodan",
  "RAG_K": 3,
  "RAG_K_MES": 2,
  "RAG_FETCH_K": 8,
  "RAG_SCORE_THRESHOLD": 1.5,
  "USE_MMR": true,
  "LAMBDA_MULT": 0.75,
  "MAX_INITIAL_RETRIEVAL": 8,
  "CHUNK_SIZE_ESTIMATE": 150,
  "MAX_VECTOR_CONTEXT_CHARS": 2200
}
```

## Heuristic Gates

```json
{
  "SMALL_TALK_MAX_WORDS": 8,
  "FOLLOWUP_RAG_MAX_WORDS": 12
}
```

## Model/Context Sanity

```json
{
  "CHECK_MODEL_CONTEXT": true,
  "AUTO_ADJUST_MODEL_CONTEXT": true
}
```

## Response Fallback Text

```json
{
  "EMPTY_STREAM_FALLBACK": "I am unable to produce a visible response right now. Please try again.",
  "QUALITY_FALLBACK_RESPONSE": "I will not repeat myself. Ask your question with more specificity."
}
```

## Notes

- `MAX_HISTORY_TURNS` is validated at startup and written back into in-memory config.
- `RAG_SCORE_THRESHOLD` is used only when `USE_MMR` is disabled.
- `RAG_K_MES` applies to the `_mes` collection path.
