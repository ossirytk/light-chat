# Config Files Documentation

Last verified: 2026-03-06

This section documents configuration files in `configs/`.

## Files

1. `configs/config.v2.json` — runtime + model + RAG settings
2. `configs/conversation_template.json` — non-mistral prompt template

## Runtime Loading Behavior

- Runtime requires `configs/config.v2.json`.
- `core/config.py` flattens nested v2 keys into legacy-style runtime keys for internal use.
- `ConversationManager` and script CLIs consume typed values via `load_conversation_runtime_config` / `load_rag_script_config`.

## Related Code

- `core/config.py`
- `core/conversation_manager.py`
- `scripts/rag/*.py`
