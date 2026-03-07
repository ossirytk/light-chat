# Conversation Quality — Current State and Next Work

Last verified: 2026-03-07

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
- Long-history summarization is implemented:
  - older turns are compacted into deterministic summary entries once a threshold is exceeded,
  - summaries are injected into prompt/history context,
  - summary behavior is configurable via `context.history.summarization` settings.
- Explicit topic-shift annotations are implemented in summarized history entries when adjacent summarized turns diverge lexically.
- Summarization regression coverage is isolated in `tests/test_history_summarization.py`; response stream/cleanup gating stays in `tests/test_response_processing.py`.
- Dynamic context is enabled by default (`USE_DYNAMIC_CONTEXT: true`).

## Not Implemented Yet

- Persona drift scoring/reporting across long sessions.
- Full offline conversation regression harness with quality metrics.

## Current Quality-Relevant Defaults

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
    }
  },
  "generation": {
    "max_stream_chars": 800,
    "max_silent_stream_chars": 120
  },
  "fallback": {
    "empty_stream": "I am unable to produce a visible response right now. Please try again.",
    "quality": "I will not repeat myself. Ask your question with more specificity."
  }
}
```

## Suggested Next Steps

1. Add a lightweight persona consistency metric in test automation.
2. Add deterministic prompt-response regression checks for release confidence.
3. Add offline summary-quality checks (coverage/faithfulness) to the regression harness.

## Related Files

- `core/conversation_manager.py`
- `core/context_manager.py`
- `tests/test_response_processing.py`
- `tests/test_history_summarization.py`
