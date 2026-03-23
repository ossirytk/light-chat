# RAG Workflows and Troubleshooting

Last verified: 2026-03-12

## Recommended End-to-End Workflow

### 1) Analyze and generate metadata

```bash
uv run python -m scripts.rag.analyze_rag_text analyze rag_data/new_char.txt -o rag_data/new_char.json --strict --review-report rag_data/new_char_review.json
```

### 2) Validate metadata

```bash
uv run python -m scripts.rag.analyze_rag_text validate rag_data/new_char.json
```

### 3) Optional quality gates

```bash
uv run python -m scripts.rag.manage_collections coverage score \
  --metadata-file rag_data/new_char.json \
  --source-file rag_data/new_char.txt \
  --threshold 0.75

uv run python -m scripts.rag.manage_collections lint message-examples --fix
```

### 4) Push lore and message examples

```bash
uv run python -m scripts.rag.push_rag_data rag_data/new_char.txt -c new_char -w
uv run python -m scripts.rag.push_rag_data rag_data/new_char_message_examples.txt -c new_char_mes -w
```

### 5) Quick retrieval spot-check

```bash
uv run python -m scripts.rag.manage_collections test new_char -q "origin" -k 5
```

### 6) Fixture-based regression run

```bash
uv run python -m scripts.rag.manage_collections evaluate-fixtures \
  --fixture-file tests/fixtures/retrieval_fixtures.json \
  --output-json logs/retrieval_eval.json \
  --output-csv logs/retrieval_eval_cases.csv \
  --history-csv logs/retrieval_eval_history.csv
```

If needed, pin the embedding model/device used by evaluation:

```bash
uv run python -m scripts.rag.manage_collections evaluate-fixtures \
  --fixture-file tests/fixtures/retrieval_fixtures.json \
  --embedding-model sentence-transformers/all-mpnet-base-v2 \
  --embedding-device cpu
```

For paired similarity vs runtime benchmarking, rerank gate checks, and interpretation guidance, see:

- `docs/rag_management/03_MANAGE_COLLECTIONS.md` → **Benchmarking Instructions (Recommended Workflow)**

## Fixture Authoring Tips

- Keep fixture IDs stable and descriptive.
- Use short, realistic queries representative of production prompts.
- Add at least one high-signal expected snippet per case.
- Prefer specific snippets over generic terms to reduce false positives.

## Common Issues

### `ModuleNotFoundError: No module named 'core'`

Cause: running nested script path directly from a context that breaks package imports.

Fix:

- Prefer module invocation for subpackages:
  - `uv run python -m scripts.rag.manage_collections ...`
- Or use top-level wrappers in `scripts/*.py`.

### No retrieval hits in `evaluate-fixtures`

Checks:

- collection exists in Chroma storage
- fixture `collection` names match actual collection names
- `k` large enough for expected snippet ranking
- embedding cache/model environment is valid

### `incompatible embedding fingerprint` errors

Cause: collection metadata indicates a different embedding model/normalization/dimension than the current run.

Fix options:

- Re-run using matching embedding flags (`--embedding-model`, `--embedding-device`).
- If collections are legacy and missing/old metadata, run fingerprint migration:

```bash
uv run python -m scripts.rag.manage_collections backfill-embedding-fingerprint --dry-run
uv run python -m scripts.rag.manage_collections backfill-embedding-fingerprint
```

- Use `--force` only when intentionally replacing conflicting fingerprint keys.

### Metadata not enriching chunks

Checks:

- metadata JSON filename matches base text name
- metadata entries contain `uuid` + text field
- text values actually appear in source chunks (exact substring matching)

### Overwrite confusion

For ingestion scripts, `--overwrite` deletes and recreates target collection. Use only when intentional.

## Operational Guidance

- Treat `analyze -> validate -> optional gates -> push -> test` as the routine workflow.
- Use module invocation in scripts under `scripts.rag` to avoid package-import surprises.
- Keep fixture history (`--history-csv`) under `logs/` and commit only if your workflow expects versioned benchmark baselines.
- Use strict metadata generation in production corpora to minimize noisy filter keys.
- Reserve `old_prepare_rag.py` for bulk rebuilds; use `push_rag_data.py` for routine updates.
