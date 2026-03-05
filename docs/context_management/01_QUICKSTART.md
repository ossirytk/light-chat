# Context Management Quickstart

Last verified: 2026-03-01

## 1) Check defaults

`configs/appconf.json` currently enables dynamic behavior:

```json
{
  "USE_DYNAMIC_CONTEXT": true,
  "MIN_HISTORY_TURNS": 2,
  "MAX_HISTORY_TURNS": 10,
  "RESERVED_FOR_RESPONSE": 384,
  "CHUNK_SIZE_ESTIMATE": 150,
  "MAX_INITIAL_RETRIEVAL": 8,
  "MAX_VECTOR_CONTEXT_CHARS": 2200
}
```

## 2) Run chat

CLI:

```bash
uv run python main.py
```

TUI:

```bash
uv run python chat_tui.py
```

## 3) Optional debug toggles

```json
{
  "DEBUG_CONTEXT": true,
  "DEBUG_PROMPT": true,
  "DEBUG_PROMPT_FINGERPRINT": true
}
```

## 4) Quick checks

- Ask a small-talk prompt (`"hi"`) and confirm retrieval is often skipped.
- Ask lore-specific prompts and confirm richer context is used.
- Verify no prompt-overflow crashes during long exchanges.

## 5) Disable dynamic mode (fallback)

```json
{
  "USE_DYNAMIC_CONTEXT": false
}
```

When disabled, retrieval still runs but without dynamic token allocation.
