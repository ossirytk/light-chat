# push_rag_data.py

Last verified: 2026-03-12

Script path: `scripts/rag/push_rag_data.py`

## Purpose

Pushes one text file into a selected Chroma collection with optional metadata enrichment.

## Command

```bash
uv run python scripts/rag/push_rag_data.py rag_data/shodan.txt -c shodan -w
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
6. Build embeddings and write to Chroma collection
7. Stamp collection-level embedding fingerprint metadata (`embedding:model`, `embedding:normalize`, `embedding:dimension` when inferable)

## Notes on Enrichment

- Metadata enrichment is parallelized using `ProcessPoolExecutor`.
- Worker processes use the `spawn` multiprocessing context to avoid Python 3.13 `fork()` deprecation warnings in multithreaded runs.
- Matched entries are injected as `document.metadata[uuid] = value`.
- If metadata file is absent, push continues without enrichment.

## Safe Operation Modes

- Use `--dry-run` to preview execution.
- Use `--overwrite` when intentionally replacing an existing collection.
- Without `--overwrite`, write attempts now validate existing collection fingerprint metadata and refuse mixed-model writes.

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
uv run python scripts/rag/push_rag_data.py rag_data/shodan.txt -c shodan -w
```

### Push with explicit embedding override

```bash
uv run python scripts/rag/push_rag_data.py rag_data/shodan.txt -c shodan -w \
   --embedding-model sentence-transformers/all-mpnet-base-v2 \
   --embedding-device cpu
```

### Push message-example collection

```bash
uv run python scripts/rag/push_rag_data.py rag_data/shodan_message_examples.txt -c shodan_mes -w
```
