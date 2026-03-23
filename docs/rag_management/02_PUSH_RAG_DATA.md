# push_rag_data.py

Last verified: 2026-03-12

Script path: `scripts/rag/push_rag_data.py`

## Purpose

Pushes one text file into a selected Chroma collection with optional metadata enrichment.

## Command

```bash
uv run python -m scripts.rag.push_rag_data rag_data/shodan.txt -c shodan -w
```

Required:

- positional `file_path`
- `--collection-name, -c`

Optional:

- `--persist-directory, -p`
- `--key-storage, -k`
- `--metadata-file, -m`
- `--chunk-size, -cs`
- `--chunk-overlap, -co`
- `--threads, -t`
- `--embedding-model`
- `--embedding-device`
- `--dry-run, -d`
- `--overwrite, -w`

## Processing Pipeline

1. Load text via `TextLoader`
2. Strip leading HTML comment header (if present)
3. Chunk text using `RecursiveCharacterTextSplitter`
4. Resolve metadata file:
   - explicit `--metadata-file`, or
   - auto-detect from file stem (`<name>.json`, with `_message_examples` normalized)
5. Enrich chunk metadata by matching known key values in chunk text
6. If metadata exists, run the source-coverage quality gate
7. Build embeddings and write to Chroma collection
8. Stamp collection-level embedding fingerprint metadata (`embedding:model`, `embedding:normalize`, `embedding:dimension` when inferable)

## Notes on Enrichment

- Metadata enrichment is parallelized using `ProcessPoolExecutor`.
- Worker processes use the `spawn` multiprocessing context to avoid Python 3.13 `fork()` deprecation warnings in multithreaded runs.
- Matched entries are injected as `document.metadata[uuid] = value`.
- If metadata file is absent, push continues without enrichment.

## Safe Operation Modes

- Use `--dry-run` to preview execution.
- Use `--overwrite` when intentionally replacing an existing collection.
- Without `--overwrite`, write attempts now validate existing collection fingerprint metadata and refuse mixed-model writes.

## Pre-Push Checklist

Before using this script for production data:

1. Generate or update `rag_data/<name>.json` with `analyze_rag_text`.
2. Validate the metadata JSON with `analyze_rag_text validate`.
3. Optionally lint `*_message_examples.txt` files with `manage_collections lint message-examples --fix`.
4. Prefer a dry run or explicit coverage review when iterating on a new corpus.

## Embedding Fingerprint and Compatibility

- Fingerprint metadata is written at collection level for newly created/overwritten collections.
- This metadata is used by runtime and collection tooling to block mixed embedding model usage.
- For legacy collections that predate fingerprint metadata, use:

```bash
uv run python -m scripts.rag.manage_collections backfill-embedding-fingerprint --dry-run
```

Then apply updates:

```bash
uv run python -m scripts.rag.manage_collections backfill-embedding-fingerprint
```

## Example Workflows

### Push lore collection

```bash
uv run python -m scripts.rag.push_rag_data rag_data/shodan.txt -c shodan -w
```

### Push with explicit embedding override

```bash
uv run python -m scripts.rag.push_rag_data rag_data/shodan.txt -c shodan -w \
   --embedding-model sentence-transformers/all-mpnet-base-v2 \
   --embedding-device cpu
```

### Push message-example collection

```bash
uv run python -m scripts.rag.push_rag_data rag_data/shodan_message_examples.txt -c shodan_mes -w
```

## Category Threshold Note

`--category-confidence-threshold` and `--allow-unassigned-categories` are exposed on the push CLI for visibility, but category assignment is determined when metadata is generated. If you want different category behavior, re-run `analyze_rag_text` to regenerate the metadata file before pushing.
