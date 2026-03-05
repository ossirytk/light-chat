# TUI Future Work — Current Scope

Last verified: 2026-03-01

The Textual UI (`chat_tui.py`) is currently stable with streaming output, status line updates, and a sidebar mode toggle.

## Already Implemented

- Two-pane layout with character card + chat area.
- Streaming assistant output into live message widget.
- Input lock while generation is active.
- Sidebar mode toggle (`F2`) between metadata and summary.
- Responsive refresh on terminal resize.
- Shared backend behavior with `ConversationManager`.

## Open Improvements

1. Add slash commands (`/clear`, `/reload`, `/help`).
2. Add session save/load support.
3. Add optional retrieval debug panel (active collection, retrieved chunk count).
4. Add message copy/export helpers.
5. Add keyboard navigation for previous prompts.

## Non-Goals for Now

- No separate web frontend.
- No additional theme system beyond Textual defaults in current app.

## Related Files

- `chat_tui.py`
- `main.py`
- `core/conversation_manager.py`
