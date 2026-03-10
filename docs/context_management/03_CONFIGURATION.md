# Context Management Configuration

Last verified: 2026-03-07

This page lists context-related keys currently used in runtime code.

## Main Toggles

```json
{
  "context": {
    "dynamic": {"enabled": true}
  },
  "debug": {
    "context": false,
    "prompt": true,
    "prompt_fingerprint": true
  }
}
```

## Budget and History

```json
{
  "context": {
    "budget": {"reserved_for_response": 384},
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
    }
  },
  "generation": {
    "max_stream_chars": 800,
    "max_silent_stream_chars": 120
  }
}
```

## Retrieval Controls

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
    "rerank": {
      "enabled": true,
      "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
      "top_n": 8
    },
    "telemetry": {"enabled": false}
  },
  "context": {
    "retrieval": {
      "max_initial_retrieval": 8,
      "chunk_size_estimate": 150,
      "max_vector_context_chars": 2200
    }
  }
}
```

## Heuristic Gates

```json
{
  "heuristics": {
    "small_talk_max_words": 8,
    "followup_rag_max_words": 12
  }
}
```

## Model/Context Sanity

```json
{
  "model": {
    "context": {
      "check": true,
      "auto_adjust": true
    }
  }
}
```

## Response Fallback Text

```json
{
  "fallback": {
    "empty_stream": "I am unable to produce a visible response right now. Please try again.",
    "quality": "I will not repeat myself. Ask your question with more specificity."
  }
}
```

## Notes

- `MAX_HISTORY_TURNS` is validated at startup and written back into in-memory config.
- History summarization compacts older turns when the threshold is exceeded and preserves the most recent turns in full.
- Summary entries include topic-shift annotations when adjacent summarized turns have no lexical overlap.
- `RAG_SCORE_THRESHOLD` is used only when `USE_MMR` is disabled.
- `RAG_K_MES` applies to the `_mes` collection path.
- Reranking is controlled by `rag.rerank.enabled` and uses `rag.rerank.top_n` candidate expansion.
