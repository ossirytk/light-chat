# Copilot Agent Prompts — Maintenance Templates

Last verified: 2026-03-07

Use these prompts in Copilot Agent mode for repeatable repo maintenance.

## Prompt 1 — Refresh docs to code reality

```
Audit all markdown files in docs/ against current code in core/, scripts/, configs/, and tests/. Update docs to match current behavior and defaults. Keep `docs/future_work/COPILOT_COMPACT_REFERENCE.md` as the single implemented-state source and `docs/future_work/REFINEMENTS.md` as the single future-work source. Mark historical content explicitly as archival pointers. Keep diffs minimal and deterministic. Validate by checking command examples still exist and config keys referenced are present in configs/*.json.
```

## Prompt 2 — Validate RAG workflow docs after script changes

```
Compare docs/RAG_SCRIPTS_GUIDE.md and `docs/future_work/COPILOT_COMPACT_REFERENCE.md` against scripts/rag/*.py and core/conversation_manager.py. Update CLI options, defaults, and examples to match actual Click commands and current config. Move any open items into `docs/future_work/REFINEMENTS.md`.
```

## Prompt 3 — Keep context-management docs synchronized

```
Cross-check docs/context_management/*.md against core/context_manager.py and dynamic retrieval logic in core/conversation_manager.py. Ensure default values and fallback behavior match current code paths and configs/config.v2.json.
```

## Post-run checks

```bash
uv run ruff check .
```

If the agent touches Python files, also run:

```bash
uv run ruff format .
uv run ruff check .
```
