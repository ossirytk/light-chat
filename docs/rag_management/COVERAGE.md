# Source Document Coverage Quality Gate

Last updated: 2026-03-16

## Overview

Coverage scoring measures what fraction of source text is represented in metadata entities. This helps identify incomplete or misaligned metadata before pushing to ChromaDB. It serves as a quality gate that can block low-quality pushes.

## Coverage Metric

**Definition**: The ratio of source text "covered" by metadata entity matches.

Specifically:
- Parse source document into characters
- For each metadata entity, find all substring matches in source (case-insensitive, fuzzy matching)
- Mark matched character ranges as "covered"
- Report: covered_chars / total_chars

**Example**:
```
Source: "Leonardo da Vinci invented the printing press and studied anatomy."
Metadata: {text: "Leonardo da Vinci"}, {text: "printing press"}

Coverage:
├─ "Leonardo da Vinci" → found at position 0-16 (covered)
├─ "printing press" → found at position 39-52 (covered)
└─ Unmapped text: "invented the... and studied anatomy."

Result: 37 covered chars / 60 total = 61.7% coverage
```

## Quality Gate Workflow

### 1. Standalone Coverage Check

```bash
# Score a metadata file against source document
uv run python -m scripts.rag.manage_collections coverage score \
  --metadata-file rag_data/shodan.json \
  --source-file rag_data/shodan.txt \
  --threshold 0.75 \
  --output-json logs/shodan_coverage.json
```

**Output**:
```
======================================================================
RAG SOURCE COVERAGE ANALYSIS
======================================================================
Entities found: 42
Source chars total: 15234
Source chars covered: 11425
Coverage ratio: 74.9%
Status: ✗ FAIL (threshold: 75%)

Category breakdown:
  faction          15 items
  technology        8 items
  location         12 items
  event             7 items

Top 5 unmapped text segments:
  - "Tri-Optimum's corporate structure changed..."
  - "The research division expanded to include..."
  - ...

Entities not found in source (4/42):
  - "Sentryman Protocol"
  - "The Hacker"
  - ...
======================================================================
```

### 2. Integration with Push Workflow

Coverage check **automatically runs** before push (can be disabled with `--force-low-coverage`):

```bash
# Push fails if coverage < threshold (default 0.75)
uv run python -m scripts.rag.push_rag_data rag_data/shodan.txt \
  -c shodan_collection \
  -m rag_data/shodan.json
  
# Output (if coverage too low):
# ✗ ERROR: Source coverage 61.3% below threshold 75%.
#          Pass --force-low-coverage to override.

# Push anyway (not recommended; for testing only)
uv run python -m scripts.rag.push_rag_data rag_data/shodan.txt \
  -c shodan_collection \
  -m rag_data/shodan.json \
  --force-low-coverage
```

## Threshold Tuning

### Default: 75%

Catches obviously incomplete metadata without being overly strict. Good for:
- Production pushes (want high confidence in coverage)
- Regular collection updates (metadata usually mature)

### Stricter: 85-90%

Use when:
- Creating new character data (validate coverage before release)
- High standards for retrieval completeness
- Can afford more metadata refinement time

Example:
```bash
uv run python -m scripts.rag.manage_collections coverage score \
  --metadata-file rag_data/new_character.json \
  --source-file rag_data/new_character.txt \
  --threshold 0.90
```

### Looser: 50-65%

Use when:
- Rapid prototyping (focus on entity extraction, not coverage)
- Iterating on metadata (coverage will improve)
- Can tolerate some unmapped content

Example:
```bash
uv run python -m scripts.rag.push_rag_data rag_data/draft.txt \
  -c draft_collection \
  -m rag_data/draft.json \
  --coverage-threshold 0.60
```

## Workflow: Improving Coverage

1. **Initial analysis**: Run coverage score, review unmapped segments
   ```bash
   uv run python -m scripts.rag.manage_collections coverage score \
     --metadata-file rag_data/shodan.json \
     --source-file rag_data/shodan.txt
   ```

2. **Identify gaps**: Review "unmapped text segments" and "entities not found"
   - Unmapped segments reveal source content without corresponding metadata entries
   - Not-found entities suggest metadata that doesn't match source text

3. **Refine metadata**: Add missing entities or fix existing ones
   ```json
   [
     {"uuid": "123e...", "text": "Tri-Optimum", "aliases": ["TriOp"], "category": "faction"},
     // Add entities for frequently-appearing unmapped segments
     {"uuid": "456f...", "text": "Hacker culture", "category": "concept"},
   ]
   ```

4. **Re-score**: Verify improvements
   ```bash
   uv run python -m scripts.rag.manage_collections coverage score \
     --metadata-file rag_data/shodan.json \
     --source-file rag_data/shodan.txt
   ```

5. **Push when ready**: Once coverage >= threshold, push to ChromaDB
   ```bash
   uv run python -m scripts.rag.push_rag_data rag_data/shodan.txt \
     -c shodan_collection \
     -m rag_data/shodan.json
   ```

## When Coverage Is Low

### Common Causes

| Cause | Symptom | Fix |
|-------|---------|-----|
| Incomplete entity extraction | Large unmapped segments | Add missing entities to metadata |
| Typos in metadata | "Entity not found" list | Fix spelling/exact text to match source |
| Too-strict threshold | Moderate coverage (60-70%) flagged as fail | Lower threshold if acceptable |
| Aliases missing | Entities not found even though mentioned | Add aliases/alternate forms |
| Format differences | "Shodan" in source, "SHODAN" in metadata | Normalize capitalization |

### When to Force Override

`--force-low-coverage` bypasses the gate. Use **only if**:
- Threshold is too strict for your use case
- You've manually reviewed unmapped content and it's acceptable
- Development/testing (not production)

**Example**:
```bash
# You've reviewed and confirmed coverage is acceptable despite low %, 
# or you want to iterate quickly
uv run python -m scripts.rag.push_rag_data rag_data/draft.txt \
  -c draft_collection \
  -m rag_data/draft.json \
  --force-low-coverage
```

## Integration with Unified Quality Gate

Coverage check is part of the upcoming unified quality-gate command:

```bash
uv run python -m scripts.rag.manage_collections quality-gate \
  --coverage-threshold 0.75 \
  --fail-on-coverage \
  --message-lint-pattern 'rag_data/*_examples.txt' \
  --fail-on-message-lint
```

This will run coverage, linting, and retrieval benchmarks in sequence, providing a single pass/fail for all RAG quality metrics.

## JSON Output Format

When `--output-json` is used:

```json
{
  "entities_count": 42,
  "source_coverage_ratio": 0.749,
  "total_source_chars": 15234,
  "covered_chars": 11425,
  "category_distribution": {
    "faction": 15,
    "technology": 8,
    "location": 12,
    "event": 7
  },
  "unmapped_sample_count": 6,
  "pass": false,
  "threshold": 0.75
}
```

Use for:
- CI/CD reporting
- Tracking coverage trends over time
- Automated decision-making (e.g., block merge if coverage fails)

## Limitations & Future Work

- **Fuzzy matching only**: Uses substring matching with 0.8 similarity threshold; does not do semantic matching
- **Character-level granularity**: Counts covered characters; doesn't weight by content importance
- **No entity de-duplication**: If metadata has similar/overlapping entities, coverage may be artificially high
- **Future**: Semantic coverage scoring, importance weighting, entity overlap detection
