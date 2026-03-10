# Context Management Quickstart

Last verified: 2026-03-06

## 1) Check defaults

`configs/config.v2.json` currently enables dynamic behavior:

```json
{
  "context": {
    "dynamic": {"enabled": true},
    "budget": {"reserved_for_response": 384},
    "history": {"min_turns": 2, "max_turns": 10},
    "retrieval": {
      "chunk_size_estimate": 150,
      "max_initial_retrieval": 8,
      "max_vector_context_chars": 2200
    }
  },
  "rag": {
    "k": 3,
    "k_mes": 2,
    "fetch_k": 8,
    "use_mmr": true,
    "lambda_mult": 0.75,
    "rerank": {"enabled": true, "top_n": 8},
    "telemetry": {"enabled": false}
  }
}
```

## 2) Run chat

CLI:

```bash
uv run python main.py
```

Web:

```bash
uv run uvicorn web_app:app --host 127.0.0.1 --port 8000
```

## 3) Optional debug toggles

```json
{
  "debug": {
    "context": true,
    "prompt": true,
    "prompt_fingerprint": true
  }
}
```

## 4) Quick checks

- Ask a small-talk prompt (`"hi"`) and confirm retrieval is often skipped.
- Ask lore-specific prompts and confirm richer context is used.
- Verify no prompt-overflow crashes during long exchanges.

## 5) Disable dynamic mode (fallback)

```json
{
  "context": {
    "dynamic": {"enabled": false}
  }
}
```

When disabled, retrieval still runs but without dynamic token allocation.
