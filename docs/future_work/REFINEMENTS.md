# Refinements Backlog

Last updated: 2026-04-03

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

### 6. Memory & Session Continuity

Inspired by analysis of Claude's leaked markdown-based memory system. The current RAG pipeline is
entirely character-centric (lore and style). There is no persistent cross-session user memory —
facts about the user, relationship state, or conversation history carry zero weight between sessions.

A two-tier hybrid is the right approach:

- **Tier 1 — Markdown persona memory (do first):** Small per-user file
  (`memory/<char_name>/<user_id>.md`, ~200–400 tokens) containing relationship facts, user
  preferences, and conversation state. Written by the LLM at session end via a lightweight
  summarisation prompt. Loaded at session start and injected into the context budget before RAG
  content. Human-readable, editable, and debuggable without any vector tooling.
- **Tier 2 — RAG over conversation archives (later):** When a user accumulates many sessions,
  semantic search over past conversations using a `<user_id>_memory` ChromaDB collection. Reuses
  the existing retrieval pipeline. Only worthwhile at scale.

Prerequisites before Tier 1 can be built:
  1. User/session identity scoping (who is this user across sessions?).
  2. Session-end write hook (trigger point for LLM memory extraction).
  3. Reserved context budget slot (~300 tokens, injected before RAG).
  4. Memory write prompt (instructs the LLM to extract 5–10 facts from the session).

Idle-time memory consolidation: merge/summarise older notes when count exceeds threshold
(equivalent to AutoDream consolidation in the Claude spec). New commands:
`/memory list`, `/memory add <note>`, `/memory forget <id>`, `/memory clear`.

Web UI: memory panel showing injected facts per turn (see `UI_REFINEMENTS.md §A.5`).

### 7. Chat Experience & Conversation Control

- **Conversation branching:** `/fork` to snapshot current state to a named branch, `/forks` to
  list all saved branches, `/fork restore <id>` to rewind and continue from that point. Branches
  stored in session JSON under a `branches` key; pairs with web UI controls (see `UI_REFINEMENTS.md §A.3`).
- **Character hot-reload:** `/character <card_name>` to swap the active character card mid-session
  without a full restart. Preserve conversation history; reset persona drift state and reload the
  RAG collection. Insert a visible "Character switched → <name>" marker in the conversation.
  List available cards with `/character list`.
- **Stop hooks:** user-defined stop conditions in config (`generation.stop_hooks`) — regex patterns
  or keyword lists with `stop | redirect | warn` actions. Useful for OOC marker detection,
  character-break detection, or content policy enforcement. Log stop events to telemetry and web
  diagnostics.
- **User-defined command macros (skills):** define custom `/skill` commands in
  `configs/skills.json`, mapping names to message templates injected at send time. Support template
  variables: `{{char}}`, `{{user}}`, `{{last_response}}`. Commands: `/skill list`,
  `/skill add <name> <template>`, `/skill remove <name>`. Web UI: skills dropdown in the chat
  input area (see `UI_REFINEMENTS.md §A.6`).

All features should be opt-in via config and must not alter existing behaviour when disabled.

### 8. Token & Context Observability

- **Pressure-aware context compaction:** replace threshold-only history summarisation with
  continuous token fill-rate tracking. Trigger compaction when the context window exceeds ~80%
  capacity. Compress oldest history segments first; keep recent turns verbatim. Emit visible
  compaction markers in the web chat UI. Expose compaction stats in the diagnostics panel.
  Builds on the existing `context_manager.py` token budget logic — low-risk addition.
- **Per-turn token usage stats:** track prompt tokens, completion tokens, context window %, and
  RAG chunk count per turn. Add session-level cumulative totals to export metadata and the ZIP
  bundle. CLI verbose mode: print token counts after each response. Web UI: extend the diagnostics
  panel (see `UI_REFINEMENTS.md §A.1–A.2`).

### 9. CLI Quality of Life

- **Output themes & syntax highlighting:** configurable terminal colour themes (dark, light,
  minimal, retro). Style character name, user input, system messages, and warnings with distinct
  ANSI colours. Store preference as `ui.cli_theme` in config. Show theme options in `/help`.
- **Customisable keybindings:** load from `configs/keybindings.json`; allow remapping of clear,
  reload, save, export, continue, and help actions. Sensible defaults matching current behaviour.
  Show current bindings in `/help` output.

### 10. Multi-Character Conversation Mode (Exploratory)

Two simultaneous active characters (e.g., narrator + character, or character A ↔ character B).
Each character maintains its own RAG collection and persona drift tracker. A turn-router
(rule-based or LLM-directed) decides which character responds each turn. Config:
`multi_character: { enabled: true, characters: ["CharA", "CharB"] }`.

**Large effort, Medium value.** Treat as a long-horizon milestone — do not start until §6–8 are
stable. The only "Large" effort item in this backlog.

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
10. Add pressure-aware context compaction and per-turn token usage stats (§8).
11. Implement Tier 1 markdown persona memory (§6) — requires user identity scoping first.
12. Add conversation branching, character hot-reload, stop hooks, and skills macros (§7).
13. CLI quality-of-life pass: themes and keybindings (§9).
14. Multi-character conversation mode (§10) — long-horizon, after §6–8 are stable.

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
