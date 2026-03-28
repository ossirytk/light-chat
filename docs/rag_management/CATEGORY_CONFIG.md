# Category Confidence Thresholds Configuration

Last updated: 2026-03-16

## Overview

Entity categories (faction, location, technology, etc.) are inferred using heuristic rules. Each inference has an associated confidence score (0.0 to 1.0). The category confidence threshold controls whether to accept or reject inferred categories below a certain confidence level.

This document explains how to configure thresholds per analysis run.

## Confidence Threshold Concept

### How Category Inference Works

For each entity, the system infers:
1. **Category**: Predicted type (faction, location, technology, character, date, event, concept)
2. **Confidence**: How certain the inference is (0.0-1.0)

### Threshold Application

- **Confidence >= threshold** → Accept the inferred category
- **Confidence < threshold AND allow_unassigned_categories=False** → Fallback to general category (usually "concept")
- **Confidence < threshold AND allow_unassigned_categories=True** → Mark category as null (unassigned)

### Example

```
Entity: "Sentryman Protocol"
Inferred category: "technology"
Confidence: 0.81 (quite confident)

With threshold 0.75 (default):
  ✓ PASS: Use category "technology" (0.81 >= 0.75)

With threshold 0.90 (strict):
  ✗ FAIL: Confidence too low (0.81 < 0.90)
  → If allow_unassigned_categories=False: use fallback "concept"
  → If allow_unassigned_categories=True: set category=null
```

## Configuration Methods

### Method 1: Default Behavior

Use the default threshold of **0.75** (recommended for most cases):

```bash
# Analyze with default threshold
uv run python -m scripts.rag.analyze_rag_text analyze rag_data/shodan.txt \
  -o rag_data/shodan.json

# Push with metadata enriched at default threshold
uv run python -m scripts.rag.push_rag_data rag_data/shodan.txt \
  -c shodan_collection \
  -m rag_data/shodan.json
```

### Method 2: Custom Threshold (Analysis)

Override threshold when analyzing *.txt files to generate metadata:

```bash
# Stricter threshold (0.85) - only assign high-confidence categories
uv run python -m scripts.rag.analyze_rag_text analyze rag_data/leonardo.txt \
  -o rag_data/leonardo.json \
  --category-confidence-threshold 0.85

# Looser threshold (0.60) - assign categories more liberally
uv run python -m scripts.rag.analyze_rag_text analyze rag_data/draft.txt \
  -o rag_data/draft.json \
  --category-confidence-threshold 0.60
```

### Method 3: Unassigned Categories

Allow entities below threshold to have `category=null`:

```bash
# Generate metadata with unassigned categories for low-confidence entities
uv run python -m scripts.rag.analyze_rag_text analyze rag_data/shodan.txt \
  -o rag_data/shodan.json \
  --category-confidence-threshold 0.85 \
  --allow-unassigned-categories
```

Result (sample):
```json
[
  {"uuid": "123e4567...", "text": "Shodan", "category": "character", "confidence": 0.94},
  {"uuid": "234f5678...", "text": "Tri-Op", "category": null, "confidence": 0.62},
  {"uuid": "345g6789...", "text": "Research Pod", "category": "technology", "confidence": 0.88}
]
```

### Method 4: Re-Analyze Before Push

Category assignment happens when metadata is generated, not when chunks are pushed to ChromaDB. If you want different category thresholds, regenerate the metadata file first and then push that result:

```bash
uv run python -m scripts.rag.analyze_rag_text analyze rag_data/shodan.txt \
  -o rag_data/shodan.json \
  --category-confidence-threshold 0.85 \
  --allow-unassigned-categories

uv run python -m scripts.rag.push_rag_data rag_data/shodan.txt \
  -c shodan_collection \
  -m rag_data/shodan.json
```

## Threshold Selection Guide

### Default: 0.75

**When to use**: Most cases

**Characteristics**:
- Balanced between strictness and coverage
- Catches clearly low-confidence inferences
- ~15-20% of inferences typically fall below threshold
- Suitable for production data

**Example**:
```bash
uv run python -m scripts.rag.analyze_rag_text analyze rag_data/shodan.txt \
  -o rag_data/shodan.json
# No threshold flag → uses 0.75 default
```

### Strict: 0.85-0.95

**When to use**: 
- High-stakes deployments (need maximum confidence in metadata)
- Creating new character data (want near-certain categorizations)
- Retrieval quality is critical
- Can tolerate larger fraction of unassigned categories

**Characteristics**:
- Only accept high-confidence inferences
- ~30-40% of inferences fall below threshold
- More unassigned categories (require manual review/fallback)
- Better quality but fewer automated assignments

**Example**:
```bash
# New character launch: high quality bar
uv run python -m scripts.rag.analyze_rag_text analyze rag_data/new_character.txt \
  -o rag_data/new_character.json \
  --category-confidence-threshold 0.90 \
  --allow-unassigned-categories

# Review and manually assign categories for null entries
# Then push when satisfied
```

### Loose: 0.50-0.65

**When to use**:
- Rapid prototyping
- Iterative development
- Quantity over quality (want most entities categorized)
- Can refine categories later via manual review

**Characteristics**:
- Accept lower-confidence inferences
- ~5-10% of inferences fall below threshold
- Most entities get assigned categories
- Higher false-positive rate (wrong categories)

**Example**:
```bash
# Draft analysis: focus on extraction, refine categories later
uv run python -m scripts.rag.analyze_rag_text analyze rag_data/draft.txt \
  -o rag_data/draft.json \
  --category-confidence-threshold 0.50

# Review output, improve metadata, then re-analyze with stricter threshold
```

## Workflow: Tuning Categories

### Phase 1: Initial Generation (Loose)

```bash
# Extract with lower confidence bar to see all inferences
uv run python -m scripts.rag.analyze_rag_text analyze rag_data/shodan.txt \
  -o rag_data/shodan_draft.json \
  --category-confidence-threshold 0.60 \
  --review-report logs/shodan_review.json
```

Check `logs/shodan_review.json` to see confidence scores for all inferences.

### Phase 2: Review & Calibrate

Look at the distribution of confidence scores:
```bash
jq '.[] | .category.confidence' logs/shodan_review.json | sort
```

This helps you choose an appropriate threshold based on your data.

### Phase 3: Final Analysis (Calibrated)

Re-run with selected threshold:
```bash
# If distribution shows clear bimodal pattern with gap at 0.78:
uv run python -m scripts.rag.analyze_rag_text analyze rag_data/shodan.txt \
  -o rag_data/shodan.json \
  --category-confidence-threshold 0.78 \
  --review-report logs/shodan_review_final.json
```

### Phase 4: Manual Refinement (Optional)

If using `--allow-unassigned-categories`, manually assign categories for null entries:

```json
// Before: low-confidence entries are null
{"uuid": "...", "text": "Some Entity", "category": null}

// After: reviewed and assigned
{"uuid": "...", "text": "Some Entity", "category": "faction"}
```

## Threshold Impact on Retrieval Quality

Stricter thresholds affect retrieval in two ways:

### 1. Ground Truth Quality

**Higher threshold** → More certain categories → Better retrieval signal
- "faction" label is reliable → Queries about factions retrieve relevant results
- Fewer ambiguous categorizations → Less noise in metadata filtering

**Lower threshold** → More covered entities → Better coverage
- More entities have categories → More retrieval opportunities
- But some categories may be incorrect → Potential false positives

### 2. Tuning for Your Use Case

Test both thresholds on your fixtures:

```bash
# Score retrieval with strict categories (0.85)
uv run python -m scripts.rag.analyze_rag_text analyze rag_data/shodan.txt \
  -o rag_data/shodan_strict.json \
  --category-confidence-threshold 0.85

# Push and evaluate retrieval
uv run python -m scripts.rag.push_rag_data rag_data/shodan.txt \
  -c shodan_strict \
  -m rag_data/shodan_strict.json \
  --overwrite

# Run retrieval benchmarks
uv run python -m scripts.rag.manage_collections evaluate-fixtures \
  --collection shodan_strict \
  --fixture tests/fixtures/retrieval_fixtures.json

# Compare results → informsthreshold choice
```

## Interaction with Strict Mode

The `--strict` flag has a different meaning:

```bash
# --strict: Use BOTH category AND alias thresholds
uv run python -m scripts.rag.analyze_rag_text analyze rag_data/shodan.txt \
  -o rag_data/shodan.json \
  --strict  # Applies threshold to both categories and aliases

# --category-confidence-threshold: Only affects category threshold
uv run python -m scripts.rag.analyze_rag_text analyze rag_data/shodan.txt \
  -o rag_data/shodan.json \
  --category-confidence-threshold 0.85  # Only categories
```

They can be combined:

```bash
# Strict on aliases + custom threshold on categories
uv run python -m scripts.rag.analyze_rag_text analyze rag_data/shodan.txt \
  -o rag_data/shodan.json \
  --strict \
  --category-confidence-threshold 0.80
```

## Metadata Format

Categories appear in generated metadata with optional confidence tracking:

```json
{
  "uuid": "12f1c8a2-af81-42ad-9b6a-48a5340fece9",
  "text": "Shodan",
  "category": "character",
  "?aliases": ["the AI", "SHODAN System"]  // Optional
}
```

**Note**: Confidence scores are only in review reports, not in final metadata JSON. If you need to track confidence, use `--review-report` output.

## Integration with Unified Quality Gate

Category configuration will be part of upcoming unified quality-gate:

```bash
uv run python -m scripts.rag.manage_collections quality-gate \
  --category-confidence-threshold 0.80 \
  --allow-unassigned-categories \
  --fail-on-coverage 0.75 \
  --fail-on-message-lint
```

## Recommendations

1. **Start with default (0.75)** for new projects
2. **Run analysis with `--review-report`** to understand distribution
3. **Calibrate threshold** based on your dataset's confidence patterns
4. **Use `--allow-unassigned-categories`** if you plan to manually review/assign
5. **Test retrieval quality** with different thresholds to find sweet spot
6. **Document your choice** in project README or CI configuration

## Troubleshooting

### All entities getting "concept" category

**Symptom**: Most entities assigned "concept" despite looking like locations/factions

**Causes**:
- Threshold too strict (increase it)
- No context keywords in source text (reword source or add aliases)
- Entity names ambiguous (add hints via metadata comments)

**Fix**:
```bash
# Loosen threshold
uv run python -m scripts.rag.analyze_rag_text analyze rag_data/file.txt \
  -o rag_data/file.json \
  --category-confidence-threshold 0.65
```

### Incorrect categories assigned

**Symptom**: Entities getting wrong categories (e.g., faction labeled as technology)

**Causes**:
- Threshold too loose (decrease it)
- Heuristic rules don't match your data patterns
- Entity text too ambiguous

**Fix**:
```bash
# Stricter threshold
uv run python -m scripts.rag.analyze_rag_text analyze rag_data/file.txt \
  -o rag_data/file.json \
  --category-confidence-threshold 0.85

# Manually review and correct in metadata JSON
```

### Too many unassigned categories

**Symptom**: Many `category: null` entries after using `--allow-unassigned-categories`

**Causes**:
- Threshold too strict
- Source data lacks context keywords

**Fix**:
```bash
# Loosen threshold or disable unassigned mode
uv run python -m scripts.rag.analyze_rag_text analyze rag_data/file.txt \
  -o rag_data/file.json \
  --category-confidence-threshold 0.70  # More inclusive

# Or allow fallback to concept category (the default)
uv run python -m scripts.rag.analyze_rag_text analyze rag_data/file.txt \
  -o rag_data/file.json \
  --category-confidence-threshold 0.85
  # Without --allow-unassigned-categories, low-confidence become "concept"
```
