# Web App Future Work — Current Scope

Last verified: 2026-03-07

The web app (`web_app.py`) is the primary interactive UI for chat and should remain aligned with `ConversationManager` behavior.

## Already Implemented

- Browser-based chat interface served by FastAPI + Jinja2 + HTMX.
- Streaming assistant output with status updates.
- Timeout handling with retry affordance in UI.
- Shared backend behavior with `ConversationManager`.
- Health diagnostics endpoints for runtime checks.

## Open Improvements

1. Add session save/load support in the web flow.
2. Add retrieval debug panel (collection, retrieved chunk count, rerank info).
3. Add message copy/export helpers.
4. Add slash-command equivalents via UI actions (`clear`, `reload`, `help`).
5. Add keyboard usability enhancements for prompt history and quick actions.

## Non-Goals for Now

- No separate TUI implementation.
- No additional frontend stack migration beyond current FastAPI + templates + HTMX.

## Related Files

- `web_app.py`
- `main.py`
- `core/conversation_manager.py`
- `templates/index.html`
- `templates/chat_message_pair.html`
