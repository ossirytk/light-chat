# analyze_rag_text.py

Last verified: 2026-03-06

Script path: `scripts/rag/analyze_rag_text.py`

## Purpose

Analyzes text corpora and produces metadata JSON suitable for RAG filtering and entity-aware retrieval.

## Commands

### 1) `analyze`

Analyze a single text file and optionally write metadata and enrichment review files.

Example:

```bash
uv run python scripts/rag/analyze_rag_text.py analyze rag_data/shodan.txt -o rag_data/shodan.json --strict --review-report rag_data/shodan_review.json
```

Options:

- `--output, -o`: output metadata JSON path
- `--min-freq, -f`: key phrase minimum frequency (default `3`)
- `--auto-categories/--no-auto-categories`: include inferred `category`
- `--auto-aliases/--no-auto-aliases`: include alias variants
- `--max-aliases`: max aliases per entity (default `5`)
- `--strict`: keep only high-confidence enrichments
- `--review-report`: write per-candidate enrichment decision log
- `--verbose, -v`: print additional analysis details

### 2) `validate`

Validate existing metadata file structure and detect common issues.

Example:

```bash
uv run python scripts/rag/analyze_rag_text.py validate rag_data/shodan.json
```

Validation checks:

- JSON parseability
- list format or `{ "Content": [...] }` format
- `uuid` presence
- text field presence (`text`, `text_fields`, `text_field`, `content`, `value`, or equivalent string field)
- duplicate UUIDs

### 3) `scan`

Scan a directory for text/metadata coverage; optionally auto-generate missing metadata files.

Example:

```bash
uv run python scripts/rag/analyze_rag_text.py scan rag_data/ --auto-generate --strict
```

Options:

- `--auto-generate, -g`: generate missing `<base>.json`
- `--auto-categories/--no-auto-categories`
- `--auto-aliases/--no-auto-aliases`
- `--max-aliases`
- `--strict`

## Output Model

Generated metadata entries are shaped like:

```json
{
  "uuid": "...",
  "text": "SHODAN",
  "category": "technology",
  "aliases": ["Sentient Hyper-Optimized Data Access Network"]
}
```

`category` and `aliases` are optional depending on flags and confidence thresholds.

## Heuristic Highlights

- Named entities: capitalization, quoted strings, date-like patterns
- Category inference uses entity/context hints (location, faction, technology, event, date, character)
- Alias generation supports parenthetical aliases, acronym extraction, and normalization variants
- Strict mode suppresses low-confidence category/alias enrichments

## Best Practices

- Use `--review-report` when tuning enrichment quality.
- Run `validate` before pushing metadata into ingestion pipelines.
- Keep `--strict` enabled for high precision corpora; disable for exploratory runs.
