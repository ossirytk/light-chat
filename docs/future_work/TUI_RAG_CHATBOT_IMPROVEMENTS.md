# TUI Future Work: RAG-Powered Character AI Chatbot

This document outlines possible future improvements for the terminal UI experience of a RAG-powered character chatbot.

## 1. Layout and Responsiveness

- Add adaptive breakpoints for narrow terminals (stack sidebar above chat instead of side-by-side).
- Provide optional collapsible sidebar (`Tab` toggle) to maximize chat space.
- Add explicit min/max widths for metadata panels to avoid truncation.
- Improve resize behavior for wrapped messages (reflow + preserve scroll anchor).

## 2. Chat Experience

- Add message timestamps and optional relative-time display.
- Add typing indicator / streaming cursor while model output is in progress.
- Add command palette shortcuts (`/clear`, `/save`, `/reload`, `/rag`, `/help`).
- Add copy/export for selected messages.

## 3. RAG Visibility and Debugging

- Add optional “RAG Inspector” panel showing:
  - active collection,
  - matched metadata keys,
  - retrieved chunk count,
  - top chunk previews.
- Add per-response toggle to display retrieved context snippets.
- Add debug mode to show whether strict metadata enrichment affected filtering.

## 4. Character and Model Context

- Add model runtime stats in sidebar (context window, temperature, top_p, kv quantization).
- Add quick character profile card with personality tags and greeting examples.
- Add command to switch character/model config without restarting TUI.

## 5. Reliability and UX Safeguards

- Add graceful cancellation of active generation (`Esc` / `Ctrl+C`).
- Add retry-on-error action for failed model responses.
- Add input history navigation (`Up`/`Down`) and draft preservation.
- Add overflow-safe rendering for long unbroken tokens.

## 6. Session Management

- Save/load named chat sessions from disk.
- Add auto-save and restore last session.
- Add metadata tagging per session (character, model, date, topic).

## 7. Accessibility and Theming

- Add built-in high-contrast and low-brightness themes.
- Add configurable color/style tokens for user vs assistant messages.
- Improve keyboard-only navigation and focus cues.

## 8. Performance

- Add batched UI updates for streaming text to reduce redraw overhead.
- Add configurable max message retention in live view with lazy history loading.
- Add profiling mode for render latency and token throughput.

## Suggested Phased Rollout

1. **Phase 1 (UX baseline):** responsive layout + sidebar toggle + command shortcuts.
2. **Phase 2 (RAG observability):** RAG Inspector and retrieval debug overlays.
3. **Phase 3 (session tooling):** save/load sessions + character/model switching.
4. **Phase 4 (polish):** themes, accessibility improvements, and performance tuning.
