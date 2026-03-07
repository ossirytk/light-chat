# Web App Future Work — Current Scope

Last verified: 2026-03-07

The web app (`web_app.py`) is the primary interactive UI for chat and should remain aligned with `ConversationManager` behavior.

## Already Implemented

- Browser-based chat interface served by FastAPI + Jinja2 + HTMX.
- Streaming assistant output with status updates.
- Timeout handling with retry affordance in UI.
- Shared backend behavior with `ConversationManager`.
- Health diagnostics endpoints for runtime checks.
- Session save/load support in the web flow.
- Retrieval debug panel (collection, chunk count, rerank summary).
- Message copy/export helpers (copy last, export TXT/JSON).
- UI equivalents for slash commands (`clear`, `reload`, `help`).
- Keyboard usability enhancements (prompt history and quick shortcuts).
- VS Code task and debug workflow for one-click start/stop.

## Open Improvements

1. Add explicit in-UI session picker (not just load-latest) with naming.
2. Add per-turn retrieval trace history (instead of latest-only panel).

## Non-Goals for Now

- No separate TUI implementation.
- No additional frontend stack migration beyond current FastAPI + templates + HTMX.

## Related Files

- `web_app.py`
- `main.py`
- `core/conversation_manager.py`
- `templates/index.html`
- `templates/chat_message_pair.html`
- `.vscode/tasks.json`
- `.vscode/launch.json`
