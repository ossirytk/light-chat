# RAG Workflows and Troubleshooting

Last verified: 2026-03-07

## Recommended End-to-End Workflow

### 1) Analyze and generate metadata

```bash
uv run python scripts/rag/analyze_rag_text.py analyze rag_data/new_char.txt -o rag_data/new_char.json --strict --review-report rag_data/new_char_review.json
```

### 2) Validate metadata

```bash
uv run python scripts/rag/analyze_rag_text.py validate rag_data/new_char.json
```

### 3) Push lore and message examples

```bash
uv run python scripts/rag/push_rag_data.py rag_data/new_char.txt -c new_char -w
uv run python scripts/rag/push_rag_data.py rag_data/new_char_message_examples.txt -c new_char_mes -w
```

### 4) Quick retrieval spot-check

```bash
uv run python -m scripts.rag.manage_collections test new_char -q "origin" -k 5
```

### 5) Fixture-based regression run

```bash
uv run python -m scripts.rag.manage_collections evaluate-fixtures \
  --fixture-file tests/fixtures/retrieval_fixtures.json \
  --output-json logs/retrieval_eval.json \
  --output-csv logs/retrieval_eval_cases.csv \
  --history-csv logs/retrieval_eval_history.csv
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

### Metadata not enriching chunks

Checks:

- metadata JSON filename matches base text name
- metadata entries contain `uuid` + text field
- text values actually appear in source chunks (exact substring matching)

### Overwrite confusion

For ingestion scripts, `--overwrite` deletes and recreates target collection. Use only when intentional.

## Operational Guidance

- Keep fixture history (`--history-csv`) under `logs/` and commit only if your workflow expects versioned benchmark baselines.
- Use strict metadata generation in production corpora to minimize noisy filter keys.
- Reserve `old_prepare_rag.py` for bulk rebuilds; use `push_rag_data.py` for routine updates.
