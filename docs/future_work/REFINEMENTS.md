# Refinements Backlog

Last updated: 2026-03-16

This is the single source for remaining and future work across quality, retrieval, and web app behavior.

## Recently Completed (2026-03-12)

- Added explicit embedding model configuration across runtime and ingestion (`embedding.model` + CLI overrides).
- Added collection-level embedding fingerprint metadata (`embedding:model`, `embedding:normalize`, `embedding:dimension`).
- Added mixed-model safety checks that block incompatible collection reads/writes.
- Added legacy fingerprint migration command: `backfill-embedding-fingerprint` (supports `--dry-run`, `--pattern`, `--force`).
- Updated RAG documentation for embedding overrides, safety checks, and migration workflow.

## Recently Completed (2026-03-16)

**Conversation Quality:**
- Added runtime persona drift scoring with hybrid metrics (heuristic + semantic-style trigram similarity).
- Added conversation-state persistence for drift telemetry (`persona_drift_history`, `persona_drift_last`, rolling average export).
- Exposed persona drift telemetry in web debug endpoints and retrieval turn traces.
- Added offline conversation quality evaluation harness: `uv run python -m scripts.conversation.evaluate_quality evaluate-conversation-fixtures`.
- Added deterministic `mock` mode and optional `live` mode for fixture evaluation.
- Added JSON/CSV/history output support and baseline comparison with soft-fail thresholds.
- Added starter conversation fixtures and automated tests for scorer determinism and evaluator behavior.

**RAG Data Quality (Metadata & Embeddings):**
- Added source-document coverage scoring (`scripts.rag.analyze_rag_coverage`) to measure what fraction of source text is represented in metadata. Push gate blocks below 0.75 by default.
- Added message examples linting (`scripts.rag.lint_message_examples`) to enforce consistent sectioning style (header presence, `[USER]:`/`[ASSISTANT]:` labels, blank line separation) across all `*_message_examples.txt` files.
- Made category confidence thresholds configurable via CLI flags (`--category-confidence-threshold`, `--allow-unassigned-categories`) on `analyze` and `push` commands.
- Added embedding model benchmarking harness (`scripts.rag.benchmark_embedding_models`) — builds ephemeral in-memory reproductions of existing collections with candidate models and compares Recall@k, MRR, MAP@k on retrieval fixture cases.
- Added re-embedding migration workflow (`scripts.rag.migrate_collection_embedding`) with atomic alias-swap (zero-downtime switchover): builds temp collection → validates against fixture (optional) → deletes old → renames temp.
- Added comprehensive test coverage (32 tests) for coverage scoring, linting, category thresholds, and all features verified ruff-clean.

## Remaining Work (Previously Scoped)

### Conversation Quality

- Calibrate drift thresholds and score weights from real session logs across at least 2 personas.
- Expand fixture coverage with `hard` and `negative` packs focused on drift, style breaks, and user-turn leakage.
- Integrate conversation quality command into a single quality-gate workflow with retrieval and RAG-data checks.
- Add CI regression policy for conversation quality baselines (warn vs hard fail by severity).
- Add docs for fixture authoring rules and baseline refresh workflow.

### RAG Data Quality

*(All scoped items completed 2026-03-16 — see Recently Completed section above.)*

### Retrieval Quality

- Add pass/fail thresholds for hard and general fixture packs (similar to rerank runtime-win gate).
- Add a lightweight trend renderer for `logs/retrieval_eval/history.csv`.

### Web App

- No pending scoped items were recorded in the prior web app future-work document.

## Possible Avenues Of Improvement

### 1. Reliability And Regression Confidence

- Create a single `uv` command that runs conversation, retrieval, and RAG-data quality gates together.
- Fail CI on quality regressions using pinned fixture packs and deterministic seeds.
- Add a release checklist doc that references only measurable checks.

### 2. Retrieval Explainability

- Add per-turn retrieval rationale snippets (why a chunk was kept after cleanup/rerank).
- Expose score deltas before/after rerank and compression in web debug panels.
- Export retrieval traces as machine-readable JSONL for post-hoc analysis.

### 3. Persona Consistency

- Add periodic persona-anchor prompts during long sessions to reduce drift.
- Track drift indicators per turn and show rolling scores in logs.
- Build a small benchmark set focused on voice and style fidelity.

### 4. Data And Corpus Hygiene

- Add strict schema versioning for metadata JSON and migration tooling.
- Add duplicate-content detection across `rag_data` documents by semantic hash.
- Add corpus freshness checks for stale character files and missing aliases.

### 5. Performance And Cost Controls

- Introduce adaptive retrieval depth by latency budget and query type.
- Add optional response-time targets and warn when exceeded.
- Benchmark sentence compression and rerank combinations on a fixed fixture matrix.
- Add an embedding model tiering profile (`small`, `balanced`, `quality`) with measured quality/cost tradeoffs.

### 6. Web UX And Observability

- Add a compact run diagnostics panel (latency, tokens/chars, retrieval counts, guardrail triggers).
- Add saveable preset profiles for debug mode and retrieval settings.
- Add one-click export bundle for support/debug sessions.

## Suggested Execution Order

1. ✅ Add metadata coverage scoring and push-blocking quality gate. (2026-03-16)
2. ✅ Benchmark embedding model candidates and select a new default profile. (2026-03-16)
3. ✅ Add re-embedding migration with rollback-safe alias switching. (2026-03-16)
4. Add automated pass/fail gates for existing retrieval fixture packs.
5. Calibrate persona drift thresholds and score weighting from recorded sessions.
6. Add hard/negative conversation fixture packs and baseline artifacts.
7. Wire conversation fixture evaluation into unified quality-gate command and CI policy.
8. Add retrieval trend rendering and debug export artifacts.
9. Iterate on higher-level UX and explainability improvements.

## Next Steps

### Retrieval Quality (Priority 1)

1. **Pass/fail thresholds:** add hard-gate logic to retrieval fixture evaluation similar to the `--require-runtime-win` flag on `benchmark-rerank`. Define baseline thresholds for general and hard fixture packs.
2. **Trend rendering:** add a lightweight CSV trend viewer for `logs/retrieval_eval/history.csv` to surface Recall@k and MRR drift over time.
3. **Collection validation:** extend `benchmark-embedding-models` to report latency and collection size metrics alongside quality scores.

### Conversation Quality (Priority 2)

1. **Calibration pass (1-2 sessions):** run long conversations for Shodan and Leonardo, export `logs/web_sessions/*`, and tune `conversation_quality.persona_drift` thresholds/weights from observed drift distributions.
2. **Fixture expansion:** add `tests/fixtures/conversation_fixtures_hard.json` and `tests/fixtures/conversation_fixtures_negative.json` with explicit expected/forbidden assertions.
3. **Baseline capture:** generate baseline reports in mock mode and store canonical artifacts under `logs/conversation_quality/baselines/`.
4. **Soft-fail policy wiring:** define and document hard-regression limits (`--max-score-drop`, `--max-drift-increase`) for local gate and CI.

### Unified Quality Gate (Priority 3)

1. **Gate command:** add one command/task that runs conversation fixtures + retrieval fixtures + RAG-data checks (coverage gate + linting) in sequence.
2. **CI integration:** wire the unified gate into GitHub Actions with pinned fixture packs and deterministic seeds; fail on regressions.
3. **Release checklist:** document measurable quality checks required before release.
