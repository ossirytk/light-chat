# RAG Document Quality Improvement Guide

This document outlines plans and ideas for improving the quality of source documents used for Retrieval Augmented Generation (RAG) in the light-chat character AI chatbot.

## Overview

The RAG system relies on two types of source material stored in `rag_data/`:

1. **Context documents** (e.g., `shodan.txt`) — lore, background information, character history, world-building.
2. **Message example files** (e.g., `shodan_message_examples.txt`) — example dialogues demonstrating character voice and style.
3. **Metadata keyword files** (e.g., `shodan.json`) — keyword/entity lists used to filter ChromaDB queries.

Improving these inputs directly improves retrieval quality and ultimately response quality.

---

## 1. Context Document Quality

### Current State

The existing context documents (e.g., `shodan.txt`) are flat prose without explicit structure. They mix multiple topics in long paragraphs, which affects chunk boundary quality during text splitting.

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

Message examples (e.g., `shodan_message_examples.txt`) are actual quotes or dialogue lines that demonstrate the character's voice, tone, and speech patterns.

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

Metadata files (e.g., `shodan.json`) contain a flat list of `{uuid, text}` entries used to filter ChromaDB queries. The `extract_key_matches` function checks whether any of these terms appear in the user's query, then uses matching UUIDs as filters.

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
| `push_rag_data.py` | Add a `--check-quality` pre-push validation step |
| New: `lint_rag_document.py` | Document linter that checks structure, sentence length, entity consistency |

---

## See Also

- [RAG_QUALITY_IMPROVEMENT.md](RAG_QUALITY_IMPROVEMENT.md) — Improving RAG retrieval pipeline quality
- [CONVERSATION_QUALITY.md](CONVERSATION_QUALITY.md) — Improving overall conversation quality
- [RAG_SCRIPTS_GUIDE.md](RAG_SCRIPTS_GUIDE.md) — How to use the RAG management scripts
