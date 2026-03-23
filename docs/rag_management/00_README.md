# RAG Management Documentation

Last verified: 2026-03-12

This folder contains detailed, script-focused documentation for everything under `scripts/rag/`.

Use module-style invocation for active commands:

```bash
uv run python -m scripts.rag.<script_name> ...
```

Top-level wrappers in `scripts/*.py` still exist for compatibility, but the recommended workflow uses the package form above.

## Scope

- Metadata analysis and generation
- Document chunking + upload to ChromaDB
- Collection operations and retrieval evaluation
- Embedding fingerprint safeguards and legacy metadata migration
- Legacy batch ingestion flow
- Practical workflows and troubleshooting

## Documents

1. `01_ANALYZE_RAG_TEXT.md`
2. `02_PUSH_RAG_DATA.md`
3. `03_MANAGE_COLLECTIONS.md`
4. `04_OLD_PREPARE_RAG.md`
5. `05_WORKFLOWS_TROUBLESHOOTING.md`

See also `docs/QUALITY_GATE.md` for the unified quality gate reference.

## Benchmarking Quick Link

- Benchmark workflows, metric interpretation, and rerank gate usage are documented in:
	- `03_MANAGE_COLLECTIONS.md` → **Benchmarking Instructions (Recommended Workflow)**

## Quality Gate

The unified quality gate (`docs/QUALITY_GATE.md`) runs RAG lint, retrieval fixtures, and
conversation fixture evaluation in one command. See that document for the full guide.

Quick start:

```bash
uv run python -m scripts.quality_gate --skip-retrieval
```

## Script Map

- `scripts/rag/analyze_rag_text.py`: analyze/validate/scan metadata
- `scripts/rag/push_rag_data.py`: single-file push into a chosen collection
- `scripts/rag/manage_collections.py`: collection operations + fixture evaluation
- `scripts/rag/old_prepare_rag.py`: legacy directory-based batch ingestion

Top-level wrapper scripts in `scripts/*.py` still exist for compatibility, but this documentation focuses on the `scripts/rag/` implementations.

## Canonical Process

For routine RAG data management, use this process:

1. Put source lore in `rag_data/<name>.txt`.
2. Generate metadata into `rag_data/<name>.json`.
3. Validate the metadata file.
4. Optionally run quality gates:
   - coverage scoring for `<name>.txt` vs `<name>.json`
   - message-example linting for `*_message_examples.txt`
5. Push lore into collection `<name>`.
6. Push message examples into collection `<name>_mes` when present.
7. Spot-check retrieval with `test`.
8. Run `evaluate-fixtures` for regression metrics when needed.

## RAG Data Layout Conventions

- `rag_data/<name>.txt`: primary lore or source text
- `rag_data/<name>.json`: metadata generated from the primary text
- `rag_data/<name>_message_examples.txt`: message examples for voice/style retrieval
- collection `<name>`: lore collection created from `<name>.txt`
- collection `<name>_mes`: message-example collection created from `<name>_message_examples.txt`
- `rag_data/archive/`: retired or historical data not intended for routine ingestion
