# Refinements Backlog

Last updated: 2026-03-09

This is the single source for remaining and future work across quality, retrieval, and web app behavior.

## Remaining Work (Previously Scoped)

### Conversation Quality

- Add persona drift scoring and reporting for long sessions.
- Add an offline conversation regression harness with deterministic quality metrics.

### RAG Data Quality

- Add source-document coverage scoring to report how much source text is represented in metadata.
- Add a quality gate that blocks push when metadata quality is below threshold.
- Add repository-wide consistency linting for sectioning style in all `*_message_examples.txt` files.
- Make category confidence thresholds configurable via CLI flags.

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

### 6. Web UX And Observability

- Add a compact run diagnostics panel (latency, tokens/chars, retrieval counts, guardrail triggers).
- Add saveable preset profiles for debug mode and retrieval settings.
- Add one-click export bundle for support/debug sessions.

## Suggested Execution Order

1. Add automated pass/fail gates for existing fixture packs.
2. Add persona drift scoring with a simple baseline metric.
3. Add metadata coverage scoring and push-blocking quality gate.
4. Add retrieval trend rendering and debug export artifacts.
5. Iterate on higher-level UX and explainability improvements.
