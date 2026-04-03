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
- **Character name mismatch on first turn (investigate).** On the first message of a session the
  persona drift scorer may not match the character name correctly, producing an artificially high
  drift score or misfire. Two candidate causes: (1) lazy initialisation — `character_name` may be
  empty when the first `PersonaAnchor` is built, as card loading (`parse_prompt`) runs during
  `__init__` before all attributes are set; (2) mes_example normalisation — the card linter
  normalises `<USER>` / `<BOT>` markers to plain `user:` / `assistant:` format, stripping the
  original character name from example turns, which may confuse the heuristic name-match on turn 1.
  Investigation steps: add a log line in `_record_retrieval_trace` printing `character_name` and
  `drift_score` at turn 1; check whether `PersonaAnchor.character_name` is populated before the
  first call; and compare drift scores with and without mes_example injection on turn 1.

### RAG Data Quality

- **Shodan lore coverage is low.** The `rag_data/` source files for Shodan are sparse relative to the
  character's depth. Coverage analysis shows many lore topics unmapped. Work needed: expand lore files
  with canonical game text (System Shock 1 & 2 dialogue, environment descriptions, terminal messages),
  re-run lint and coverage checks, then rebuild the collection.

*(Previously scoped items completed; implemented state is tracked in `docs/future_work/COPILOT_COMPACT_REFERENCE.md`.)*

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

### 11. Character Card Import & Avatar Support

Improve the character loading pipeline to support richer card formats and give each character a
visible identity in the UI.

#### 11.1 Character Avatar / Icon Upload

Each character should have an optional avatar image displayed in the chat UI next to assistant
messages and in the character selector. Implementation:

- Store avatars in `character_storage/<character_name>/avatar.png` (or `.jpg`, `.webp`).
- Serve via `GET /characters/{name}/avatar` — returns the image, falls back to a generated
  initial/monogram placeholder if no avatar is found.
- Web UI: display avatar thumbnail in the chat header and optionally next to each assistant message
  bubble. Upload button on the character settings page (see `UI_REFINEMENTS.md §C`).
- Keep the image small (≤ 512 px, ≤ 200 KB) — resize on upload with Pillow.

#### 11.2 Character Card V2 / V3 Import

The project currently loads character data from plain JSON files in `cards/`. Extend this to
support importing from standard character card formats used by the wider AI chat ecosystem.

**Character Card V2 (PNG `chara` tEXt chunk):**
- PNG files with a `chara` tEXt chunk containing base64-encoded JSON (TavernCardV2 format).
- Fields map directly to existing config: `name`, `description`, `scenario`, `mes_example`,
  `first_mes`, `personality`, `system_prompt`, `post_history_instructions`, `character_book`.
- Already partially supported via `cards/leonardo_da_vinci.png` — formalise the import path.

**Character Card V3 (CCv3 — the current community standard):**
- Spec: <https://github.com/kwaroran/character-card-spec-v3/blob/main/SPEC_V3.md>
- PNG/APNG: JSON embedded in `ccv3` tEXt chunk as UTF-8 → base64. If both `chara` and `ccv3`
  chunks are present, prefer `ccv3`.
- CHARX: zip file with `card.json` at root. Assets (icons, backgrounds, emotion sprites) live in
  `assets/{type}/` subdirectories and can be accessed via `embeded://path` URIs.
- JSON: plain `.json` file containing the CharacterCardV3 object directly.
- V3 adds: `assets[]` (icon, background, emotion images), `nickname`, `group_only_greetings`,
  `creation_date`, `modification_date`, `source[]`, multilingual creator notes.
- **Implementation priority:** PNG V2 import is simplest and highest value (most cards in the wild
  are V2 PNG). V3 PNG import is a small additional step. CHARX support can come later.
- Use the `pypng` or `Pillow` library to read tEXt chunks; no heavyweight dependency needed.
- On import: write a normalised JSON card to `cards/` and optionally extract the embedded avatar
  to `character_storage/<name>/avatar.png`.

**Suggested implementation order:**
1. Formalise V2 PNG import (read `chara` chunk → normalise → save JSON + avatar).
2. V3 PNG import (read `ccv3` chunk → normalise; fall back to `chara` if absent).
3. Avatar display in web UI (§11.1 + `UI_REFINEMENTS.md §C`).
4. CHARX import (zip extraction + asset handling).
5. In-app card editor (§11.3 below).

#### 11.3 In-App Character Card Editor

A web-based form editor for creating and editing character cards without leaving the application.
Several community implementations can be used as reference for field layout and PNG embedding:

- [ZoltanAI/character-editor](https://github.com/ZoltanAI/character-editor) — lightweight
  browser-side editor for V1/V2 cards; entirely static HTML/JS, good reference for field layout.
- SillyTavern's built-in editor supports V2 fields and lorebook editing.
- The [CCv3 spec](https://github.com/kwaroran/character-card-spec-v3/blob/main/SPEC_V3.md)
  provides the canonical field reference for a V3-compatible editor.

**Backend requirements:**

- `GET /characters/{name}/edit` — load existing card fields into the edit form.
- `POST /characters/{name}/edit` — validate and save edited fields to `cards/<name>.json`.
- `GET /characters/new` / `POST /characters/new` — create a new card from scratch.
- `POST /characters/{name}/export/png` — embed card JSON into a PNG tEXt chunk (`ccv3`) and
  return the PNG for download. Uses the stored avatar as the base image.
- `POST /characters/{name}/avatar` — upload a new avatar image (resize to ≤ 512 px with Pillow,
  save to `character_storage/<stem>/avatar.png`). Replaces `UI_REFINEMENTS.md §C.2`.

**Field coverage (minimum viable):**
`name`, `description`, `scenario`, `personality`, `first_mes`, `mes_example`,
`voice_instructions` (project-specific), `tags`, `creator`, `system_prompt`.
Lorebook / `character_book` editing is out of scope for the initial version.

**PNG embedding:**
Read tEXt chunks with `struct` (stdlib) or `Pillow`; write `ccv3` chunk (base64-encoded UTF-8
JSON). Also write a `chara` chunk for backward compatibility with V2 readers. No new heavy
dependencies needed — `Pillow` is already a likely dependency for image resizing.

**Effort:** Medium. The form and routing are straightforward; the PNG round-trip (read → edit →
re-embed) is the only non-trivial part. Build after §11.1–11.2 so the parsing layer is shared.

## §12 User-Facing Documentation Site

A non-technical, friendly guide for people who want to use light-chat without programming knowledge,
hosted alongside the repository as a static site.

### Motivation

The web UI has grown substantially (chat, RAG management, diagnostics, session search, character
management). Many features have in-UI help text, but there is no cohesive end-user reference.
A dedicated documentation site lowers the barrier to entry and helps non-technical users
understand what the tool does and how to use it.

### Hosting options (all compatible with GitHub Pages)

| Option | Notes |
|--------|-------|
| **MkDocs Material** (recommended) | Python-based, clean modern theme, markdown source. Fits the project's Python tooling; `mkdocs gh-deploy` publishes to GitHub Pages. Add as a `uv` dev dependency. |
| **Docsify** | Single HTML file + plain markdown; zero build step, works directly from a `docs/` folder on GitHub Pages. Good for rapid publishing. |
| **Docusaurus** | Node.js/React, strong search and versioning. More setup overhead; worthwhile if the docs grow large. |

**Recommended starting point:** MkDocs Material. One `mkdocs.yml` config, `uv add --dev mkdocs-material`,
and `uv run mkdocs gh-deploy` is all that's needed. The source already lives in `docs/`.

### Content scope (minimum viable)

| Page | Audience-level description |
|------|---------------------------|
| **Welcome / What is this?** | Plain-language intro: local AI chatbot, character cards, no data sent to cloud |
| **Getting started** | How to install, configure a model, and start the server |
| **Using the chat** | Sending messages, sessions (save/load/search), keyboard shortcuts, export |
| **Character cards** | What they are, where to put card files, adding an avatar image |
| **Knowledge base (RAG)** | Plain-language: what a "collection" is, how to add a new character's info, what "coverage" means |
| **Settings and profiles** | What each retrieval setting does in plain English; saving and applying profiles |
| **Diagnostics panel** | What the token bar and per-turn table show; how to read drift scores |
| **Troubleshooting** | Common errors, model not loading, no collections found, stream timeout |

### Implementation notes

- Place MkDocs source in `docs/` (already exists) with `mkdocs.yml` at the repository root.
- Separate developer/contributor documentation (current `docs/`) from user-guide pages
  (`docs/user_guide/`) using MkDocs navigation sections.
- The in-UI help guides (chat sidebar and RAG page) can be reused or adapted as source material.
- A GitHub Actions workflow can automate `mkdocs gh-deploy` on every push to `main`.

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
11. Implement V2/V3 card import and avatar upload pipeline (§11.1–11.2).
12. Build in-app character card editor with PNG export (§11.3 + `UI_REFINEMENTS.md §C.5`).
13. Implement Tier 1 markdown persona memory (§6) — requires user identity scoping first.
14. Add conversation branching, character hot-reload, stop hooks, and skills macros (§7).
15. CLI quality-of-life pass: themes and keybindings (§9).
16. Multi-character conversation mode (§10) — long-horizon, after §6–8 are stable.
17. Publish user-facing documentation site (§12) — MkDocs Material on GitHub Pages.

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
