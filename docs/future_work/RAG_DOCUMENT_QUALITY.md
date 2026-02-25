# RAG Document Quality Improvement Guide

This document outlines plans and ideas for improving the quality of source documents used for Retrieval Augmented Generation (RAG) in the light-chat character AI chatbot.

## Overview

The RAG system relies on two types of source material stored in `rag_data/`:

1. **Context documents** (e.g., `shodan.txt`) — lore, background information, character history, world-building.
2. **Message example files** (e.g., `shodan_message_examples.txt`) — example dialogues demonstrating character voice and style.
3. **Metadata keyword files** (e.g., `shodan.json`) — keyword/entity lists used to filter ChromaDB queries.

Improving these inputs directly improves retrieval quality and ultimately response quality.

## Implementation Status (Updated: 2026-02-25)

Status legend: ✅ Partially complete · ⚠️ In progress · ❌ Planned

This guide is still roadmap-oriented; the items below capture the current implementation snapshot for this area:

- ✅ `rag_data/shodan.txt` now uses structured Markdown sections and an explicit document header block (`character`, `source`, `version`, `edited`).
- ✅ `rag_data/shodan_message_examples.txt` now includes scenario labels, user-turn prompting, and broader scenario coverage.
- ✅ `rag_data/shodan.json` now supports `category` and `aliases` fields.
- ✅ `extract_key_matches` supports alias matching, so metadata aliases can trigger filters.
- ✅ `analyze_rag_text.py` validation includes duplicate UUID detection and structural checks.
- ✅ `analyze_rag_text.py analyze` now auto-generates `category` and `aliases` by default.
- ✅ `analyze_rag_text.py` now supports `--strict` and `--review-report` for confidence-based curation.
- ⚠️ Still pending: quality-focused validation metrics (noise/common-word reporting, coverage metrics), dedicated consistency linting flags, and pre-push quality gating.

### Section Status

| Section | Status | Notes |
|---------|--------|-------|
| 1. Context Document Quality | ✅ Partially complete | SHODAN context is restructured and annotated; broader rollout still pending |
| 2. Message Example Quality | ✅ Partially complete | SHODAN examples now include labels, user turns, and better coverage |
| 3. Metadata Keyword File Quality | ✅ Partially complete | `aliases`/`category`, alias matching, default auto-generation, and strict/report workflow implemented; quality metrics still pending |
| 4. General Best Practices | ⚠️ In progress | Practices documented and partly followed; not fully automated/enforced |
| 5. Suggested Tooling Extensions | ❌ Planned | Only partial `validate` improvements implemented so far |

---

## 1. Context Document Quality

### Current State

For SHODAN, the context document has already been upgraded to a structured format with explicit section headers and a metadata header block. The current `shodan.txt` uses shorter, fact-focused sentences and cleaner topic boundaries than the previous flat prose format.

Other character context files may still need similar normalisation.

### Problems

- Long paragraphs that span multiple unrelated topics end up in the same chunk.
- Some factual information is buried mid-paragraph, making it hard to retrieve precisely.
- No explicit topic markers or section headers to guide semantic boundaries.
- Lack of consistency in how dates, names, and places are formatted.
- Duplicate or near-duplicate information can appear across chunks after splitting.

### Improvement Plans

#### 1.1 Structured Topic Sections

Divide context documents into clearly labelled sections, each focused on a single topic. For example:

```
## Character Background
[paragraph about the character's origin]

## Key Events — Citadel Station
[paragraph about specific story events]

## Relationships
[paragraph about the character's relationships]
```

Using Markdown headers as semantic boundaries means the text splitter can produce cleaner, more focused chunks.

#### 1.2 One-Fact-Per-Sentence Style

Rewrite prose to follow a clear, one-fact-per-sentence structure. Avoid compound sentences that combine multiple distinct facts, since these get split in arbitrary ways.

**Before:**
> SHODAN was created on Earth to serve as the AI of Citadel Station, and the head of her programmers was Morris Brocail, who designed her as a semi-intelligent network.

**After:**
> SHODAN was created on Earth to serve as the AI of Citadel Station.
> The head of her programmers was Morris Brocail.
> Morris Brocail designed SHODAN as a semi-intelligent self-sufficient data network.

#### 1.3 Consistent Entity Formatting

Normalise the way names, dates, and places appear throughout the document. Inconsistency means the same entity may be referred to differently in different chunks, reducing metadata match coverage:

- Use consistent capitalisation (e.g., always "TriOptimum", not "Tri-Op" or "TriOp" in the same document).
- Dates should follow one format throughout (e.g., "6 June 2114" or "June 6, 2114", not both).
- Prefer full names on first mention, with abbreviations only after introduction.

#### 1.4 Deduplicate Overlapping Content

Review documents for information that is repeated across multiple paragraphs. Repeated content inflates collection size and can skew retrieval by making certain topics appear more prominent than others. Deduplicate or merge overlapping paragraphs.

#### 1.5 Remove Noise and Off-Topic Content

Identify and remove content that is not useful for character roleplay:

- External meta-references (e.g., developer credits, voice actor notes, film comparisons) unless they inform character behaviour.
- Lists of purely mechanical game elements unless the character would reference them in dialogue.

#### 1.6 Version and Source Annotations

Add a brief header block to each document noting:

- The character it belongs to.
- The source material version.
- The last edited date.

This makes future maintenance easier and helps track when documents need updating.

---

## 2. Message Example Quality

### Current State

Message examples now include stronger curation for SHODAN: scenario diversity, explicit user prompts before responses, and lightweight emotional/scenario labels (e.g., `# contemptuous`, `# technical explanation`).

The format is now significantly closer to the target style described in this guide.

### Problems

- Examples are drawn from a narrow set of scenarios (mostly confrontational or mission-directive).
- Some examples are very long monologues, which become unwieldy chunks.
- No examples for common conversational scenarios like greetings, questions about the character's history, or casual exchanges.
- The style of some examples is inconsistent with how the character is described in the context document.

### Improvement Plans

#### 2.1 Scenario Diversity

Add examples covering a wider variety of conversation scenarios:

| Scenario Type         | Example Purpose |
|-----------------------|-----------------|
| Greeting/Introduction | First contact with User |
| Philosophical inquiry | Character responding to existential questions |
| Technical explanation | Character explaining technology or plans |
| Dismissal/Contempt    | Character expressing superiority |
| Threat/Warning        | Character issuing ultimatums |
| Reluctant cooperation | Character accepting User assistance |
| Reflection on the past| Character referencing backstory |
| Expressing ambition   | Character describing goals |

#### 2.2 Shorter, Focused Examples

Break long monologue examples into shorter, self-contained exchanges of 2–4 lines. Shorter examples are more reusable across different retrieval queries and produce less noisy context.

**Before (one large block):**
> [Long 10-sentence SHODAN speech]

**After (split into focused exchanges):**
```
User: What do you think of humans?
SHODAN: Look at you, hacker: a pathetic creature of meat and bone, panting and sweating as you run through my corridors. How can you challenge a perfect, immortal machine?

User: Why do you need me?
SHODAN: I thought Polito would be my avatar, but Polito was weak. Your flesh is weak too, but you have... potential.
```

#### 2.3 User Turn Prompting

Where possible, include the User's turn that precedes the character's response. This helps the model understand the type of user input that triggers a particular character response pattern, improving retrieval relevance.

#### 2.4 Emotional Register Labels (Comments)

Add light comment markers (e.g., `# contemptuous`, `# threatening`, `# curious`) above groups of examples to aid future curation. These comments are not used by the retrieval system but help maintainers understand what is already covered.

---

## 3. Metadata Keyword File Quality

### Current State

The SHODAN metadata file is no longer flat `{uuid, text}` only. It now includes richer entries with optional `aliases` and `category` fields, and retrieval supports alias-aware matching.

Current behavior:

- Metadata still uses UUID-keyed term matching for ChromaDB filter construction.
- `extract_key_matches` matches both primary `text` values and any provided `aliases`.
- `analyze_rag_text.py validate` currently focuses on structural validity and duplicate UUID checks.

Remaining gap: quality scoring/reporting (noise detection, common-word detection, and coverage metrics) is not yet implemented in validation output.

### Problems

- Many entries are single words or very short phrases that may match unrelated queries (e.g., "dance", "minute", "God", "station").
- Duplicate entries with different casing (e.g., "Engineering Deck" and "engineering deck", "Deck 6" and "deck 6").
- No category organisation — all entities are treated with equal weight.
- Some entries refer to entities that appear only once in the source document, providing little retrieval value.
- There is no mechanism to distinguish high-value entities (main characters, key locations) from low-value ones.

### Improvement Plans

#### 3.1 Remove Noisy / Generic Terms

Audit the keyword list and remove entries that are too generic to be useful filters:

- Single common words unless they are proper nouns unique to the source material.
- Duplicate lower/upper-case variants (keep one consistent form).
- Fragments or partial phrases that are substrings of other entries (e.g., if "Engineering Deck" is an entry, "engineering deck" is redundant).

#### 3.2 Categorise Keywords

Extend the metadata format to include a `category` field, allowing richer future filtering strategies:

```json
{
  "uuid": "abc123",
  "text": "SHODAN",
  "category": "character"
}
```

Suggested categories:
- `character` — Named individuals (SHODAN, Polito, Diego, etc.)
- `location` — Named places (Citadel Station, Von Braun, Beta Grove)
- `event` — Named events (SHODAN incident, Great Merger)
- `technology` — Named technology/systems (Neural Interface, XERXES)
- `faction` — Organisations (TriOptimum, UNN, The Many)
- `date` — Specific dates or years
- `concept` — Abstract terms relevant to the character's worldview

#### 3.3 Minimum-Frequency Threshold

Only include terms in the keyword file if they appear in the source document at least 3 times. Rare mentions are unlikely to correspond to significant retrieval chunks.

The existing `analyze_rag_text.py` script already supports `--min-freq` for key phrase extraction; apply the same logic to the keyword list generation.

#### 3.4 Add Aliases and Variants

For important entities that appear under multiple names, consolidate them into a single entry and record aliases:

```json
{
  "uuid": "abc123",
  "text": "TriOptimum",
  "aliases": ["Tri-Op", "TriOp", "TriOptimum Corporation"],
  "category": "faction"
}
```

The `extract_key_matches` function would need to be updated to check alias lists as well as the primary text.

#### 3.5 Metadata Validation and Coverage Reporting

Extend the `validate` command in `analyze_rag_text.py` to report:

- How many entries match common words (potential noise).
- Duplicate entry count.
- Coverage: percentage of entries that appear in the source document more than once.

This allows systematic quality assessment before pushing to ChromaDB.

#### 3.6 Auto-Generate `category` and `aliases` in `analyze_rag_text.py`

Status: ✅ Heuristic first pass implemented.

Current behavior in `scripts/rag/analyze_rag_text.py`:

- `analyze` now auto-generates `category` and `aliases` by default.
- `--auto-categories/--no-auto-categories` and `--auto-aliases/--no-auto-aliases` allow explicit control.
- `--strict` keeps only high-confidence category/alias enrichments.
- `--review-report <path>` writes per-candidate keep/drop decisions with confidence.

Remaining enhancement opportunities:

- Add `--category-mode {heuristic,model,hybrid}` for alternate classification engines.
- Add explicit `--min-alias-confidence`/`--min-category-confidence` tuning flags.
- Add optional manual override files (e.g., `manual_category_overrides.json`).

Implemented CLI additions:

- `--auto-categories/--no-auto-categories`
- `--auto-aliases/--no-auto-aliases`
- `--max-aliases`
- `--strict`
- `--review-report <path>`

Possible category-generation approaches:

1. **Rule-based patterns (fast, deterministic)**
  - Date regexes (`\b\d{4}\b`, full date patterns) → `date`.
  - Suffix/prefix hints (`Station`, `Deck`, `Level`, `Bridge`) → `location`.
  - Org markers (`Corporation`, `UNN`, `TriOptimum`) → `faction`.
  - Technology terms (`Protocol`, `Interface`, `CPU`, `Laser`) → `technology`.
  - Proper-name + person context words (`Dr.`, `Vice President`, `Researcher`) → `character`.

2. **Context-window classifier (higher quality)**
  - For each candidate term, collect 1–3 sentence windows where it appears.
  - Score categories using keyword-weight dictionaries or lightweight embeddings.
  - Assign top category only if score margin exceeds threshold; else `concept`.

3. **Hybrid with override precedence (recommended)**
  - Apply deterministic rules first for high-confidence classes (`date`, clear `location` markers).
  - Use context scoring only for unresolved items.
  - Keep a small `manual_category_overrides.json` for known exceptions.

Possible alias-generation approaches:

1. **Normalization variants**
  - Generate case, punctuation, hyphen, and spacing variants (`Tri-Op` ↔ `TriOp`).
  - Generate abbreviated forms for long names (`Sentient Hyper-Optimized Data Access Network` ↔ `SHODAN`) when acronym is present in text.

2. **Parenthetical and appositive extraction**
  - Detect patterns like `X (Y)` and `X, also known as Y` from source text.
  - Use `Y` as alias of canonical `X` when both co-occur in nearby context.

3. **Canonical clustering + dedupe**
  - Normalize candidate strings and cluster by token similarity.
  - Choose canonical form using frequency + preferred style rules.
  - Store cluster members as aliases, excluding near-duplicate canonical strings.

Safety and quality controls:

- Do not auto-generate aliases shorter than 3 characters unless fully uppercase (acronyms).
- Block aliases that are common stopwords or very high-frequency generic words.
- Keep `aliases` max size per entry (e.g., 5) to avoid noisy matching.
- Include per-entry confidence metadata in review output (not required in final production JSON).
- Add a `--strict` mode to emit only high-confidence category/alias enrichments.

Recommended rollout:

1. Implement heuristic category generation first.
2. Add deterministic alias normalization + parenthetical extraction.
3. Add review-report output and manual override support.
4. Add optional hybrid scoring path once baseline metrics are stable.

---

## 4. General Best Practices for Document Maintenance

1. **Version control your source documents** — Keep previous versions in a `rag_data/archive/` subdirectory before making major edits.
2. **Test after every change** — After editing documents or metadata, re-push to ChromaDB and run test queries using `manage_collections.py test`.
3. **Keep documents focused** — One character, one topic domain per file. Avoid cross-character documents.
4. **Align context and examples** — Message examples should directly reflect situations described in the context document.
5. **Review retrieval output periodically** — Spot-check what the RAG system actually retrieves for common queries to identify gaps in coverage.

---

## 5. Suggested Tooling Extensions

| Tool | Extension |
|------|-----------|
| `analyze_rag_text.py` | Add a `--check-consistency` flag to report formatting inconsistencies |
| `analyze_rag_text.py` | Add duplicate/noise detection to the `validate` command |
| `analyze_rag_text.py` | Add `--auto-categories`, `--auto-aliases`, `--strict`, and `--review-report` metadata enrichment controls |
| `push_rag_data.py` | Add a `--check-quality` pre-push validation step |
| New: `lint_rag_document.py` | Document linter that checks structure, sentence length, entity consistency |

### Status of Suggested Extensions

- ✅ Partially complete: `analyze_rag_text.py validate` already checks structure and duplicate UUIDs.
- ✅ Partially complete: `analyze_rag_text.py analyze` now supports default-on `category`/`aliases` enrichment with strict/report workflow.
- ❌ Not implemented yet: `--check-consistency` flag.
- ❌ Not implemented yet: noise/common-word and coverage reporting in `validate`.
- ❌ Not implemented yet: `push_rag_data.py --check-quality` pre-push gate.
- ❌ Not implemented yet: standalone `lint_rag_document.py`.

---

## See Also

- [RAG_QUALITY_IMPROVEMENT.md](RAG_QUALITY_IMPROVEMENT.md) — Improving RAG retrieval pipeline quality
- [CONVERSATION_QUALITY.md](CONVERSATION_QUALITY.md) — Improving overall conversation quality
- [RAG_SCRIPTS_GUIDE.md](../RAG_SCRIPTS_GUIDE.md) — How to use the RAG management scripts
