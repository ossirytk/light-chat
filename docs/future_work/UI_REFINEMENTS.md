# UI Refinements Backlog

Last updated: 2026-04-03

Forward-looking improvements to the web interface. This covers chat UI enhancements and the larger
RAG management UI plan.

Implemented state lives in `docs/future_work/COPILOT_COMPACT_REFERENCE.md`.

---

## A. Chat UI Enhancements

Quality-of-life improvements to the main chat view and diagnostics panel. These complement the
backend features described in `docs/future_work/REFINEMENTS.md §§7–8`.

### A.1 Token Budget Visualization

Add a token budget bar to the diagnostics panel (or below the chat input area).

- Displays real-time allocation breakdown: system prompt %, history %, RAG context %, headroom.
- Updates after each turn alongside the persona drift score.
- Colour-coded: green (< 70%) / yellow (70–90%) / red (> 90%) by fill pressure.
- Data source: extend `ContextBudget` in `context_manager.py` to return percentage breakdowns;
  emit alongside each response payload or via a dedicated `/diagnostics/budget` endpoint.

### A.2 Per-Turn Token Usage Stats Panel

Extend the existing diagnostics panel with turn-level token stats.

- Show per-turn: prompt tokens, completion tokens, context window %, RAG chunks retrieved.
- Show session-level cumulative totals at the bottom of the panel.
- Backend: `context_manager.py` already computes budgets; add a `TokenUsageRecord` emitted
  alongside each streamed response.
- Depends on `REFINEMENTS.md §8` (per-turn token usage stats) for the backend data.

### A.3 Conversation Branching Controls

UI surface for the fork/restore feature described in `REFINEMENTS.md §7`.

- **Fork button** in the chat header: prompts for an optional name, then snapshots current
  conversation state as a named branch.
- **Branch list** in the Sessions panel: shows all forks for the active session with timestamps
  and restore buttons.
- **Restore action**: rewinds the chat view and conversation state to the fork point; inserts a
  visible `— Restored from fork: <name> —` divider in the chat history.
- Backend: extend `export_conversation_state` / `import_conversation_state` to support a
  `branches` key in the session JSON.
- Depends on `REFINEMENTS.md §7` (conversation branching) for the backend state model.

### A.4 Session History Search

Full-text search across saved session JSON files, surfaced in the Sessions panel.

- Search input with character name filter and optional date range.
- Results show matching turns with surrounding context snippets.
- Routes: `GET /sessions/search?q=<query>&character=<name>&from=<date>&to=<date>`.
- Returns HTMX partial with a paginated results table; consistent with existing panel pattern.

### A.5 Memory Panel

UI surface for the persistent memory system described in `REFINEMENTS.md §6`. Deferred until
Tier 1 markdown memory is implemented.

- Add a "Memory" tab in the diagnostics sidebar.
- Show which memory entries were injected for the current turn.
- If Tier 2 RAG memory is active, show relevance scores alongside each entry.
- Manual actions: add fact, forget entry (calls `/memory` endpoints).
- Routes: `GET /memory`, `POST /memory/add`, `DELETE /memory/{id}`.

### A.6 Skills / Macros Dropdown

UI surface for the skills system described in `REFINEMENTS.md §7`. Deferred until the skills
config backend is implemented.

- Skills dropdown button adjacent to the chat input field.
- Lists available `/skill` commands with their template previews; selecting one inserts the
  expanded template into the input for editing before send.
- Reflects live state from `configs/skills.json` without a page reload.

---

## B. RAG Management UI

A dedicated panel (or separate page) that exposes all data management and ChromaDB collection
operations currently only accessible via CLI scripts. The goal is full operational parity with
the `scripts/rag/` CLI toolset, accessible from the browser without a terminal.

### B.1 Scope

| Area | Features |
|------|---------|
| **RAG data files** | List, view, run linting, run coverage analysis |
| **Collections** | List, inspect, delete, rebuild, query test |
| **Fixture evaluation** | Run evaluate-fixtures, view results, view trend history |
| **Embedding benchmarking** | Trigger benchmark run, view results |
| **Collection migration** | Re-embed to new model, backfill fingerprints |

Out of scope for this plan (requires broader changes):
- In-browser text editing of `rag_data/` source files
- File upload / new character creation
- Real-time log streaming during long-running jobs (deferred to async job tracker)

### B.2 Routes

All routes under `/rag/` prefix. Backend calls existing CLI modules directly via Python
imports (no subprocess); UI responses use HTMX partial renders consistent with existing chat UI.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/rag` | RAG management root panel |
| GET | `/rag/files` | List `rag_data/` files with status badges |
| GET | `/rag/files/{filename}` | View file content (read-only) |
| POST | `/rag/lint` | Run message-example linting; return results table |
| POST | `/rag/lint/fix` | Run linting with auto-fix; return diff summary |
| POST | `/rag/coverage` | Run coverage analysis on a lore file; return score + report |
| GET | `/rag/collections` | List ChromaDB collections with counts and fingerprints |
| GET | `/rag/collections/{name}` | Collection detail: model, dimensions, sample docs |
| DELETE | `/rag/collections/{name}` | Delete collection (with confirmation step) |
| POST | `/rag/collections/{name}/query` | Ad-hoc test query; return top-k chunks with scores |
| POST | `/rag/collections/{name}/push` | Rebuild collection from source file (long-running) |
| GET | `/rag/evaluate` | Fixture evaluation index: list fixture packs |
| POST | `/rag/evaluate/run` | Run evaluate-fixtures for a given pack; return metrics table |
| GET | `/rag/evaluate/trends` | Show retrieval trend table from history CSV |
| POST | `/rag/benchmark/embedding` | Trigger embedding model benchmark (long-running) |
| GET | `/rag/benchmark/embedding/results` | View last benchmark results |
| POST | `/rag/collections/{name}/backfill-fingerprint` | Backfill fingerprint metadata |

Long-running operations (push, benchmark) use a simple in-memory job store and HTMX polling
(`hx-trigger="every 2s"`) to update a progress indicator until the job completes.

### B.3 UI Layout

**Option A — Sidebar panel (minimal change)**
Add a "RAG" tab to the existing sidebar alongside the Sessions and Debug panels. Each section
expands inline. Suitable for read-only operations and short-running commands (lint, query test,
collection list).

**Option B — Separate page `/rag` (recommended)**
A dedicated full-width management page with a left nav (Files | Collections | Evaluate | Benchmark)
and a main content area. Keeps the chat UI uncluttered. Shares the same dark theme and HTMX
stack; no new JS libraries needed.

### B.4 Templates to Create

```
templates/
  rag/
    layout.html           — outer shell with left nav, reuses base styles
    files_list.html       — table of rag_data/ files with lint/coverage badges
    file_view.html        — pre-formatted file content + action buttons
    lint_results.html     — per-file violation table (errors/warnings)
    coverage_report.html  — coverage score bar + unmatched-chunk list
    collections_list.html — collection cards (name, docs, model, actions)
    collection_detail.html — single collection stats + sample doc viewer
    query_results.html    — ranked chunk list with scores and metadata
    push_status.html      — job progress widget (polled via HTMX)
    evaluate_index.html   — fixture pack selector
    evaluate_results.html — metrics table (recall, mrr, per-case breakdown)
    trends_table.html     — run history table with delta columns
    benchmark_results.html — embedding model comparison table
```

### B.5 Backend Modules to Add

```
core/rag_manager.py       — thin façade over manage_collections + push_rag_data
                            provides synchronous calls for lint, coverage,
                            collection list/delete/info, query, evaluate
core/job_queue.py         — simple in-memory job store for long-running ops
                            (push, benchmark); tracks status + result for HTMX polling
```

Or, given the project's existing pattern, call the CLI module functions directly from
`web_app.py` route handlers (no new modules needed for the synchronous operations).

### B.6 Implementation Order

1. **Collections list + info** — read-only, safe starting point; validates routing and template
   pattern before adding mutations.
2. **Ad-hoc query test** — immediate value; requires no mutations; validates embedding model
   loading in web context.
3. **Lint run + fix** — stateless, fast; validates the partial-render / error-table pattern.
4. **Collection delete** — add confirmation modal pattern (reusable for later destructive ops).
5. **Coverage analysis** — reuses lint results pattern.
6. **Fixture evaluation + trends** — longer output; validates the results-table + history pattern.
7. **Collection push (rebuild)** — first long-running job; validates async job store + HTMX
   polling pattern.
8. **Embedding benchmark** — second long-running job; reuses job/polling pattern.
9. **Fingerprint backfill** — maintenance op; reuses collection detail + confirmation pattern.

### B.7 Dependencies and Constraints

- ChromaDB client must be initialized once and shared (existing `ConversationManager` already
  does this; a lightweight separate client init is also acceptable for the RAG panel).
- Embedding model load is expensive (~5 s cold start); cache the loaded model across requests
  in the web process (existing `ConversationManager` already owns one instance).
- Deletion and push are destructive; both require a confirmation step in the UI before the
  backend action fires.
- Push operations on large files may take 30–120 s; the HTMX polling pattern must handle
  job timeouts gracefully (show error state after N polls with no completion).
- All fixture evaluation uses `similarity` mode only in the web UI (no live ConversationManager
  retrieval) to avoid model double-loading.

### B.8 Non-Goals (Deferred)

- In-browser text editor for `rag_data/` source files (use VS Code or a dedicated CMS).
- File upload for new character data (filesystem write from web raises deployment concerns).
- Real-time log streaming for long-running jobs (stdout pipe to WebSocket — separate effort).
- Multi-user / authentication (single-user local tool assumption).

---

## Suggested Execution Order (UI)

1. RAG Management UI (§B) as a self-contained milestone — implement §B.6 steps in order.
2. Token budget visualization (§A.1) and per-turn token stats panel (§A.2) — low-risk extensions
   to the existing diagnostics panel; depends on `REFINEMENTS.md §8` backend work.
3. Session history search (§A.4) — stateless read-only feature, no new backend state model needed.
4. Conversation branching controls (§A.3) — depends on `REFINEMENTS.md §7` session state changes.
5. Memory panel (§A.5) — depends on `REFINEMENTS.md §6` Tier 1 memory being implemented first.
6. Skills dropdown (§A.6) — depends on `REFINEMENTS.md §7` skills config backend.
