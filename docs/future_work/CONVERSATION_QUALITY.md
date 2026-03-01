# Conversation Quality — Current State and Next Work

Last verified: 2026-03-01

This document tracks what is implemented in conversation quality and what remains open.

## Implemented

- Prompt guidance enforces in-character responses and explicitly disallows generating `User:` lines.
- `voice_instructions` from character cards is supported and injected into prompt assembly when present.
- Streaming safety guards are active:
  - early stop when generated user-turn markers appear,
  - max streamed output cap via `MAX_STREAM_CHARS`,
  - silent-stream guard via `MAX_SILENT_STREAM_CHARS`,
  - configurable visible fallback via `EMPTY_STREAM_FALLBACK`.
- Post-processing strips model-format artifacts and truncates generated user-turn spillover.
- Response quality gating is applied before history write:
  - rejects too-short responses,
  - rejects responses containing user-turn markers,
  - rejects exact duplicates of the previous AI turn,
  - writes `QUALITY_FALLBACK_RESPONSE` into history when a response fails checks.
- History depth is configurable and currently set to `MAX_HISTORY_TURNS: 10`.
- Dynamic context is enabled by default (`USE_DYNAMIC_CONTEXT: true`).

## Not Implemented Yet

- Long-history summarization (older turns compressed into compact summaries).
- Explicit topic-shift annotations in conversation history.
- Persona drift scoring/reporting across long sessions.
- Full offline conversation regression harness with quality metrics.

## Current Quality-Relevant Defaults

From `configs/appconf.json`:

```json
{
  "USE_DYNAMIC_CONTEXT": true,
  "MIN_HISTORY_TURNS": 2,
  "MAX_HISTORY_TURNS": 10,
  "MAX_STREAM_CHARS": 800,
  "MAX_SILENT_STREAM_CHARS": 120,
  "EMPTY_STREAM_FALLBACK": "I am unable to produce a visible response right now. Please try again.",
  "QUALITY_FALLBACK_RESPONSE": "I will not repeat myself. Ask your question with more specificity."
}
```

## Suggested Next Steps

1. Add history summarization once paired history exceeds a threshold (for example 8–10 turns).
2. Add a lightweight persona consistency metric in test automation.
3. Add deterministic prompt-response regression checks for release confidence.

## Related Files

- `core/conversation_manager.py`
- `core/context_manager.py`
- `tests/test_response_processing.py`
