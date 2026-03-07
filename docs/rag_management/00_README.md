# RAG Management Documentation

Last verified: 2026-03-07

This folder contains detailed, script-focused documentation for everything under `scripts/rag/`.

## Scope

- Metadata analysis and generation
- Document chunking + upload to ChromaDB
- Collection operations and retrieval evaluation
- Legacy batch ingestion flow
- Practical workflows and troubleshooting

## Documents

1. `01_ANALYZE_RAG_TEXT.md`
2. `02_PUSH_RAG_DATA.md`
3. `03_MANAGE_COLLECTIONS.md`
4. `04_OLD_PREPARE_RAG.md`
5. `05_WORKFLOWS_TROUBLESHOOTING.md`

## Benchmarking Quick Link

- Benchmark workflows, metric interpretation, and rerank gate usage are documented in:
	- `03_MANAGE_COLLECTIONS.md` → **Benchmarking Instructions (Recommended Workflow)**

## Script Map

- `scripts/rag/analyze_rag_text.py`: analyze/validate/scan metadata
- `scripts/rag/push_rag_data.py`: single-file push into a chosen collection
- `scripts/rag/manage_collections.py`: collection operations + fixture evaluation
- `scripts/rag/old_prepare_rag.py`: legacy directory-based batch ingestion

Top-level wrapper scripts in `scripts/*.py` still exist for compatibility, but this documentation focuses on the `scripts/rag/` implementations.
