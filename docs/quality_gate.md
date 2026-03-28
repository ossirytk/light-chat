# Quality Gate

Last verified: 2026-03-26

The unified quality gate runs all quality checks in one command and prints a PASS/WARN/FAIL summary table. It covers RAG data linting, retrieval fixture evaluation, and conversation fixture evaluation.

## Quick Start

```bash
# No ChromaDB required - suitable for local development and CI
uv run python -m scripts.quality_gate --skip-retrieval

# Full gate with live retrieval (requires ChromaDB collections)
uv run python -m scripts.quality_gate
```

## What It Checks

Steps run in sequence:

### Step 1 — RAG Data Lint

Scans `rag_data/*_message_examples.txt` for style violations using `MessageExamplesLinter`. Reports the number of errors and warnings across all files. Exits non-zero on any ERROR-level violation.

### Step 2 — Retrieval Fixtures

Evaluates three fixture packs in `similarity` mode:

| Fixture | Purpose |
|---------|---------|
| `tests/fixtures/retrieval_fixtures.json` | General regression pack |
| `tests/fixtures/retrieval_fixtures_hard.json` | Hard/paraphrase stress cases |
| `tests/fixtures/retrieval_fixtures_negative.json` | Forbidden-snippet bleed-through cases |

Pass/fail thresholds (`--min-retrieval-recall`, `--min-retrieval-mrr`) are applied to the **general pack only**. Hard and negative packs report metrics but do not gate on thresholds by default.

Use `--skip-retrieval` to bypass all three steps when ChromaDB collections are not available (e.g., fresh CI environment).

### Step 3 — Conversation Fixtures

Evaluates three conversation fixture packs in **deterministic mock mode** (no live model required):

| Fixture | Purpose |
|---------|---------|
| `tests/fixtures/conversation_fixtures.json` | Baseline Shodan cases |
| `tests/fixtures/conversation_fixtures_hard.json` | Drift, style-break, user-turn leakage (Shodan + Leonardo) |
| `tests/fixtures/conversation_fixtures_negative.json` | Template leakage, OOC phrase forbidden assertions |

If baseline artifacts exist under `logs/conversation_quality/baselines/`, the gate compares current scores against the baseline and issues a regression warning or failure when deltas exceed thresholds.

## CLI Options

```
--seed INTEGER                  Deterministic seed for mock evaluation [default: 42]
--baselines-dir PATH            Directory for conversation baseline JSON files
                                [default: logs/conversation_quality/baselines]
--max-score-drop FLOAT          Max allowed decrease in avg_turn_score [default: 0.08]
--max-drift-increase FLOAT      Max allowed increase in avg_drift_score [default: 0.08]
--min-retrieval-recall FLOAT    Min Recall@k for general retrieval pack
--min-retrieval-mrr FLOAT       Min MRR for general retrieval pack
--retrieval-history-csv PATH    Append retrieval metrics to this CSV
--conversation-history-csv PATH Append conversation metrics to this CSV
--skip-retrieval                Skip all retrieval fixture steps
--strict                        Promote WARN results to FAIL
--help
```

## Output Format

```
Running quality gate checks...

[1/3] RAG data lint
  -> FAIL  53 error(s), 0 warning(s) across 2 file(s)

[2/3] Retrieval fixtures
  retrieval_fixtures.json: PASS  cases=10 skipped=0 recall=0.900 mrr=0.850
  retrieval_fixtures_hard.json: PASS  cases=10 skipped=0 recall=0.700 mrr=0.600
  retrieval_fixtures_negative.json: PASS  cases=5 skipped=0 recall=0.000 mrr=0.000

[3/3] Conversation fixtures (mock mode)
  conversation_fixtures.json: PASS  turns=4 persona=0.510 drift=0.490 score=0.706
  conversation_fixtures_hard.json: PASS  turns=12 persona=0.526 drift=0.474 score=0.715
  conversation_fixtures_negative.json: PASS  turns=5 persona=0.514 drift=0.486 score=0.708

========================================================================
Step                                      Status  Detail
------------------------------------------------------------------------
  RAG lint: message-examples              FAIL    53 error(s)...
  Retrieval: retrieval_fixtures.json      PASS    ...
  ...
========================================================================
Gate FAILED (1 step(s)). See table above.
```

Exit codes:
- `0` — all steps passed (or warned, when `--strict` is not set)
- `1` — one or more steps failed (or warned under `--strict`)

## Baseline Capture

Before using baseline comparison, capture canonical baselines in mock mode:

```bash
uv run capture-conversation-baselines
```

This runs all three fixture files with seed 42 and writes JSON baselines to `logs/conversation_quality/baselines/`. Re-run with `--force` to overwrite after intentional score changes.

## CI Integration

`.github/workflows/quality_gate.yml` runs on every push and pull request:

1. `uv run ruff check .` — lint
2. `uv run ruff format --check .` — format check
3. `uv run pytest -q` — unit tests
4. `capture-conversation-baselines` — idempotent baseline capture (writes if missing)
5. `quality-gate --skip-retrieval` — full gate without retrieval (no ChromaDB in CI)

To add retrieval checks to CI, set up a ChromaDB collection in the workflow and remove `--skip-retrieval`.

## Retrieval Trend Tracking

After collecting a few evaluation runs to `logs/retrieval_eval/history.csv`, render a trend table:

```bash
uv run python -m scripts.rag.manage_collections show-retrieval-trends \
  --history-csv logs/retrieval_eval/history.csv \
  --last-n 20
```

Each row shows Recall@k and MRR for one run, with delta columns (`dRecall`, `dMRR`) vs the preceding row.

## Adding the Quality Gate to `[project.scripts]`

`quality-gate` and `capture-conversation-baselines` are registered in `pyproject.toml`:

```bash
uv run quality-gate --skip-retrieval
uv run capture-conversation-baselines
```

## Fixture Authoring Rules

### Retrieval fixtures (`tests/fixtures/retrieval_fixtures*.json`)

```json
{
  "collection": "shodan",
  "k": 5,
  "dashboard_ks": [1, 3, 5],
  "cases": [
    {
      "id": "unique_case_id",
      "query": "The retrieval query",
      "expected_snippets": ["snippet that must appear in results"],
      "forbidden_snippets": ["text that must NOT appear in results"],
      "min_expected_matches": 1
    }
  ]
}
```

- `expected_snippets` — at least one must appear in top-k results for the case to be a hit
- `forbidden_snippets` — if any appear in results, the case is marked `FORBIDDEN_HIT`
- `min_expected_matches` — how many expected snippets must match for a hit (default 1)

### Conversation fixtures (`tests/fixtures/conversation_fixtures*.json`)

```json
{
  "persona": "Shodan",
  "cases": [
    {
      "id": "unique_case_id",
      "persona": "Shodan",
      "turns": [
        {
          "user": "What the user says",
          "expected_contains": ["phrase that should appear in mock response"],
          "forbidden_contains": ["phrase that must NOT appear"]
        }
      ]
    }
  ]
}
```

- `expected_contains` — case-insensitive substring matches; empty list means no content check
- `forbidden_contains` — any match counts as a forbidden hit and penalizes the turn score
- Mock mode generates deterministic responses from a seed; they rarely match content assertions, so conversation fixtures primarily test schema parsing and drift scoring unless run in `live` mode

## Remaining Work

See `docs/future_work/REFINEMENTS.md` for pending items, including:
- Fixture authoring rules and baseline refresh workflow docs
- Persona drift calibration from real session logs
- Retrieval `benchmark-embedding-models` latency/collection-size reporting
