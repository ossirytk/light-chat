# manage_collections.py

Last verified: 2026-03-12

Script path: `scripts/rag/manage_collections.py`

## Purpose

Operational CLI for Chroma collections: inspection, deletion, retrieval tests, exports, and fixture-based retrieval evaluation.

Embedding fingerprint safety is enforced across retrieval/evaluation commands to prevent mixed-model reads.

## Command Group

You can run either form, but prefer the module form:

```bash
uv run python scripts/manage_collections.py --help
```

or

```bash
uv run python -m scripts.rag.manage_collections --help
```

## Commands

### 1) `list-collections`

List all collections, optionally verbose.

```bash
uv run python -m scripts.rag.manage_collections list-collections -v
```

Options:

- `--persist-directory, -p`
- `--verbose, -v`

### 2) `delete`

Delete a single collection.

```bash
uv run python -m scripts.rag.manage_collections delete shodan_old -y
```

Options:

- `--persist-directory, -p`
- `--yes, -y`

### 3) `delete-multiple`

Bulk delete with wildcard pattern.

```bash
uv run python -m scripts.rag.manage_collections delete-multiple --pattern "test_*" -y
```

Options:

- `--persist-directory, -p`
- `--pattern`
- `--yes, -y`

### 4) `test`

Ad-hoc retrieval test for one collection and query.

```bash
uv run python -m scripts.rag.manage_collections test shodan -q "SHODAN origin" -k 5
```

Behavior:

- For non-`_mes` collections, attempts metadata-key matching and staged filters (`$and`/`$or`/unfiltered)
- For each attempt, runs similarity search and prints score + content preview

Options:

- `--query, -q` (required)
- `--k`
- `--persist-directory, -p`
- `--key-storage, -k`
- `--embedding-model`
- `--embedding-device`

### 5) `export`

Export full collection payload to JSON.

```bash
uv run python -m scripts.rag.manage_collections export shodan -o backups/shodan.json
```

Options:

- `--output, -o` (required)
- `--persist-directory, -p`

### 6) `info`

Show collection stats + one sample record.

```bash
uv run python -m scripts.rag.manage_collections info shodan
```

Options:

- `--persist-directory, -p`

### 7) `evaluate-fixtures`

Run regression fixtures and compute retrieval metrics.

```bash
uv run python -m scripts.rag.manage_collections evaluate-fixtures \
  --fixture-file tests/fixtures/retrieval_fixtures.json \
  --retrieval-mode runtime \
  --show-failures
```

Hard/paraphrase stress set example:

```bash
uv run python -m scripts.rag.manage_collections evaluate-fixtures \
  --fixture-file tests/fixtures/retrieval_fixtures_hard.json \
  --show-failures
```

Negative bleed-through set example:

```bash
uv run python -m scripts.rag.manage_collections evaluate-fixtures \
  --fixture-file tests/fixtures/retrieval_fixtures_negative.json \
  --show-failures
```

Rerank-sensitive benchmark set example:

```bash
uv run python -m scripts.rag.manage_collections evaluate-fixtures \
  --fixture-file tests/fixtures/retrieval_fixtures_rerank.json \
  --retrieval-mode runtime \
  --show-failures
```

Metrics:

- `Recall@k`
- `MRR` (mean reciprocal rank)
- `ExpRecall@k` (macro expected-snippet recall)
- `Precision@k`
- `MAP@k`
- dashboard hit-rate slices from fixture `dashboard_ks` (e.g., `hit_rate_at_1`, `hit_rate_at_3`, ...)

Fixture cases can additionally define `forbidden_snippets`; if any forbidden snippet appears in results, the case is marked as `FORBIDDEN_HIT` / `forbidden_match`.

Options:

- `--fixture-file`
- `--k`
- `--retrieval-mode` (`similarity` or `runtime`)
- `--persist-directory, -p`
- `--embedding-model`
- `--embedding-device`
- `--show-failures`
- `--output-json`
- `--output-csv`
- `--history-csv` (append one summary row per run)
- `--min-recall` (exit non-zero if Recall@k falls below this threshold)
- `--min-mrr` (exit non-zero if MRR falls below this threshold)

Hard-gate example — fail CI if general pack drops below thresholds:

```bash
uv run python -m scripts.rag.manage_collections evaluate-fixtures \
  --fixture-file tests/fixtures/retrieval_fixtures.json \
  --min-recall 0.70 \
  --min-mrr 0.50
```

`runtime` mode uses `ConversationManager` retrieval (`_search_collection`), so MMR/rerank behavior is included in evaluation.

All collections are fingerprint-validated before evaluation. If a collection has conflicting fingerprint metadata, evaluation exits with a clear error.

#### Pass/fail gate usage

Exit non-zero when Recall@k falls below a threshold:

```bash
uv run python -m scripts.rag.manage_collections evaluate-fixtures \
  --fixture-file tests/fixtures/retrieval_fixtures_hard.json \
  --min-recall 0.7 \
  --min-mrr 0.5
```

Both flags are optional and independent; omitting a flag skips that threshold check.

### 8) `benchmark-rerank`

Run the rerank-sensitive fixture set in both `similarity` and `runtime` modes and print one machine-friendly delta line.

```bash
uv run python -m scripts.rag.manage_collections benchmark-rerank
```

CI gate example (non-zero exit if runtime is worse than similarity on Recall@k or MRR):

```bash
uv run python -m scripts.rag.manage_collections benchmark-rerank --require-runtime-win
```

Optional embedding override flags:

- `--embedding-model`
- `--embedding-device`

### 9) `backfill-embedding-fingerprint`

Backfill fingerprint metadata for legacy collections created before fingerprint enforcement.

Preview:

```bash
uv run python -m scripts.rag.manage_collections backfill-embedding-fingerprint --dry-run
```

Apply:

```bash
uv run python -m scripts.rag.manage_collections backfill-embedding-fingerprint
```

Target subset and force conflicts:

```bash
uv run python -m scripts.rag.manage_collections backfill-embedding-fingerprint \
  --pattern "shodan*" --force
```

Options:

- `--persist-directory, -p`
- `--pattern`
- `--embedding-model`
- `--embedding-device`
- `--force`
- `--dry-run`

### 10) `show-retrieval-trends`

Display a compact table of Recall@k and MRR over time from a retrieval eval history CSV.

```bash
uv run python -m scripts.rag.manage_collections show-retrieval-trends \
  --history-csv logs/retrieval_eval/history.csv
```

Show only the last 10 runs:

```bash
uv run python -m scripts.rag.manage_collections show-retrieval-trends \
  --history-csv logs/retrieval_eval/history.csv \
  --last-n 10
```

Output columns: run number, date, fixture file, retrieval mode, k, Recall@k, MRR, delta Recall (vs prior run), delta MRR.

Options:

- `--history-csv` (default: `logs/retrieval_eval/history.csv`)
- `--last-n` (show only the last N rows; default: all rows)

The history CSV is populated by passing `--history-csv` to any `evaluate-fixtures` run.

### Benchmarking Instructions (Recommended Workflow)

1) Generate paired reports for a fixture pack:

```bash
uv run python -m scripts.rag.manage_collections evaluate-fixtures \
  --fixture-file tests/fixtures/retrieval_fixtures_rerank.json \
  --retrieval-mode similarity \
  --output-json logs/retrieval_eval/rerank_similarity.json \
  --output-csv logs/retrieval_eval/rerank_similarity.csv \
  --history-csv logs/retrieval_eval/history.csv

uv run python -m scripts.rag.manage_collections evaluate-fixtures \
  --fixture-file tests/fixtures/retrieval_fixtures_rerank.json \
  --retrieval-mode runtime \
  --output-json logs/retrieval_eval/rerank_runtime.json \
  --output-csv logs/retrieval_eval/rerank_runtime.csv \
  --history-csv logs/retrieval_eval/history.csv
```

2) Run gate check for rerank-sensitive regressions:

```bash
uv run python -m scripts.rag.manage_collections benchmark-rerank --require-runtime-win
```

3) Interpret deltas:

- `delta_recall > 0` and `delta_mrr > 0` means runtime retrieval stack improved over baseline
- `delta_recall < 0` or `delta_mrr < 0` means regression and gate failure (when required)

## Runtime vs Similarity Benchmark Pair

Use the dedicated rerank fixture pack to measure retrieval-stack impact with minimal noise (these are the same two runs done by `benchmark-rerank`):

```bash
uv run python -m scripts.rag.manage_collections evaluate-fixtures \
  --fixture-file tests/fixtures/retrieval_fixtures_rerank.json \
  --retrieval-mode similarity

uv run python -m scripts.rag.manage_collections evaluate-fixtures \
  --fixture-file tests/fixtures/retrieval_fixtures_rerank.json \
  --retrieval-mode runtime
```

Interpretation:

- Higher runtime `MRR` indicates better top-rank ordering after runtime retrieval logic.
- Higher runtime `Recall@k` indicates runtime mode recovers more expected snippets within top-k.
- If metrics are identical, the fixture set likely does not yet stress rerank/MMR-sensitive cases enough.

Last observed on this workspace (2026-03-07):

- `similarity`: `Recall@5 = 0.667`, `MRR = 0.194`
- `runtime`: `Recall@5 = 1.000`, `MRR = 1.000`

## Report Outputs

### JSON report (`--output-json`)

Contains:

- generation timestamp
- fixture path
- `k`
- summary block (`evaluated`, `skipped`, `hits`, `recall_at_k`, `mrr`)
- summary block also includes `expected_recall_at_k`, `precision_at_k`, `map_at_k`, and dashboard hit-rate keys
- per-collection aggregate metrics under `collections`
- evaluated case list
- skipped case list

### CSV case report (`--output-csv`)

One row per case with:

- id, status, rank, hit_at_k, forbidden_hit, k, collection, query,
  min_expected_matches, expected_total, matched_expected,
  expected_recall_at_k, precision_at_k, average_precision_at_k,
  expected_snippets, forbidden_snippets, forbidden_matches

### CSV run history (`--history-csv`)

One row per run with:

- generated_at, fixture_file, k, retrieval_mode,
  evaluated, skipped, hits, recall_at_k, mrr,
  expected_recall_at_k, precision_at_k, map_at_k

## Practical Usage

```bash
uv run python -m scripts.rag.manage_collections evaluate-fixtures \
  --fixture-file tests/fixtures/retrieval_fixtures.json \
  --output-json logs/retrieval_eval.json \
  --output-csv logs/retrieval_eval_cases.csv \
  --history-csv logs/retrieval_eval_history.csv
```

### 10) `show-retrieval-trends`

Display a compact trend table from a retrieval eval history CSV, showing Recall@k and MRR per run with deltas vs the previous row.

```bash
uv run python -m scripts.rag.manage_collections show-retrieval-trends \
  --history-csv logs/retrieval_eval/history.csv
```

Limit to the most recent N rows:

```bash
uv run python -m scripts.rag.manage_collections show-retrieval-trends \
  --history-csv logs/retrieval_eval/history.csv \
  --last-n 10
```

Options:

- `--history-csv` (default: `logs/retrieval_eval/history.csv`)
- `--last-n` (default: show all rows)

The table prints columns: `#`, `Date`, `Fixture`, `Mode`, `k`, `Recall@k`, `MRR`, `dRecall`, `dMRR`.
