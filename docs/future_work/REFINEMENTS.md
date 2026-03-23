# Refinements Backlog

Last updated: 2026-03-26

This is the single source for remaining and future work across quality and retrieval.

Web UI and UX improvements are tracked separately in `docs/future_work/UI_REFINEMENTS.md`.

Implemented state lives in `docs/future_work/COPILOT_COMPACT_REFERENCE.md`.

## Remaining Work (Previously Scoped)

### Conversation Quality

- Calibrate drift thresholds and score weights from real session logs across at least 2 personas.
- ✅ Expand fixture coverage with `hard` and `negative` packs focused on drift, style breaks, and user-turn leakage. (2026-03-26)
- ✅ Integrate conversation quality command into a single quality-gate workflow with retrieval and RAG-data checks. (2026-03-26)
- ✅ Add CI regression policy for conversation quality baselines (warn vs hard fail by severity). (2026-03-26)
- Add docs for fixture authoring rules and baseline refresh workflow.

### RAG Data Quality

*(All scoped items completed; implemented state is tracked in `docs/future_work/COPILOT_COMPACT_REFERENCE.md`.)*

### Retrieval Quality

- ✅ Add pass/fail thresholds for hard and general fixture packs (similar to rerank runtime-win gate). (2026-03-26)
- ✅ Add a lightweight trend renderer for `logs/retrieval_eval/history.csv`. (2026-03-26)

### Web App

- See `docs/future_work/UI_REFINEMENTS.md` for all web UI and UX improvement plans.

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

*(Web UX and observability improvements are tracked in `docs/future_work/UI_REFINEMENTS.md`.)*

## Suggested Execution Order

1. ✅ Add metadata coverage scoring and push-blocking quality gate. (2026-03-16)
2. ✅ Benchmark embedding model candidates and select a new default profile. (2026-03-16)
3. ✅ Add re-embedding migration with rollback-safe alias switching. (2026-03-16)
4. ✅ Add automated pass/fail gates for existing retrieval fixture packs. (2026-03-26)
5. Calibrate persona drift thresholds and score weighting from recorded sessions.
6. ✅ Add hard/negative conversation fixture packs and baseline artifacts. (2026-03-26)
7. ✅ Wire conversation fixture evaluation into unified quality-gate command and CI policy. (2026-03-26)
8. ✅ Add retrieval trend rendering and debug export artifacts. (2026-03-26)
9. Iterate on higher-level UX and explainability improvements — see `docs/future_work/UI_REFINEMENTS.md`.

## Next Steps

### Retrieval Quality (Priority 1)

1. **Pass/fail thresholds:** add hard-gate logic to retrieval fixture evaluation similar to the `--require-runtime-win` flag on `benchmark-rerank`. Define baseline thresholds for general and hard fixture packs.
2. **Trend rendering:** add a lightweight CSV trend viewer for `logs/retrieval_eval/history.csv` to surface Recall@k and MRR drift over time.
3. **Collection validation:** extend `benchmark-embedding-models` to report latency and collection size metrics alongside quality scores.

### Conversation Quality (Priority 2)

1. **Calibration pass (1-2 sessions):** run long conversations for Shodan and Leonardo, export `logs/web_sessions/*`, use `calibrate-persona-drift`, and tune `conversation_quality.persona_drift` thresholds/weights from observed drift distributions.
2. **Fixture expansion:** add `tests/fixtures/conversation_fixtures_hard.json` and `tests/fixtures/conversation_fixtures_negative.json` with explicit expected/forbidden assertions.
3. **Baseline capture:** generate baseline reports in mock mode and store canonical artifacts under `logs/conversation_quality/baselines/`.
4. **Soft-fail policy wiring:** define and document hard-regression limits (`--max-score-drop`, `--max-drift-increase`) for local gate and CI.

### Unified Quality Gate (Priority 3)

1. **Gate command:** add one command/task that runs conversation fixtures + retrieval fixtures + RAG-data checks (coverage gate + linting) in sequence.
2. **CI integration:** wire the unified gate into GitHub Actions with pinned fixture packs and deterministic seeds; fail on regressions.
3. **Release checklist:** document measurable quality checks required before release.
