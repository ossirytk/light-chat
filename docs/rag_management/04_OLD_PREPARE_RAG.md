# old_prepare_rag.py (Legacy)

Last verified: 2026-03-06

Script path: `scripts/rag/old_prepare_rag.py`

## Status

Legacy batch ingestion helper. Kept for compatibility and directory-wide ingestion workflows.

Prefer `push_rag_data.py` for new, explicit single-file/collection operations.

## Purpose

Scans a documents directory and creates collections in bulk:

- `<collection>.txt` -> `<collection>`
- `<collection>_message_examples.txt` -> `<collection>_mes`

## Command

```bash
uv run python scripts/rag/old_prepare_rag.py -d rag_data/ -p character_storage/
```

Options:

- `--documents-directory, -d`
- `--persist-directory, -p`
- `--key-storage, -ks`
- `--threads, -t`
- `--chunk-size, -cs`
- `--chunk-overlap, -co`

## Behavior Summary

1. Enumerates all `*.txt` files in documents directory.
2. Treats files not ending in `_message_examples` as base collections.
3. Chunks each file with `RecursiveCharacterTextSplitter`.
4. Enriches chunks using metadata from `<base>.json` (if available).
5. Writes vectors to Chroma.
6. If `<base>_message_examples.txt` exists, writes to `<base>_mes`.

## Differences vs `push_rag_data.py`

- `old_prepare_rag.py`: implicit multi-file batch processing by naming convention
- `push_rag_data.py`: explicit one-file one-target collection operation, easier to control and reason about

## When to Use

- Migration/rebuild of many collections at once
- Legacy workflows that rely on directory conventions

## When Not to Use

- Fine-grained or CI-style ingestion where explicit collection targets are preferred
