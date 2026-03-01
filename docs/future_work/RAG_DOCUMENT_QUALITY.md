# RAG Document Quality — Current State and Next Work

Last verified: 2026-03-01

This document tracks source-data quality for files in `rag_data/`.

## Implemented

- Context and message-example files support leading header comments for document metadata.
- Ingestion now strips the leading HTML comment before embedding (`strip_leading_html_comment`), preventing metadata-header noise from entering the vector store.
- Metadata entries support richer fields (`category`, `aliases`) beyond `uuid` + text.
- Alias-aware matching is implemented in retrieval filter helpers (`extract_key_matches`).
- Metadata validation includes structural checks and duplicate UUID detection.
- Analysis supports quality controls:
  - `--auto-categories/--no-auto-categories`
  - `--auto-aliases/--no-auto-aliases`
  - `--max-aliases`
  - `--strict`
  - `--review-report`

## Remaining Gaps

- No coverage score that reports how much of each source document is represented by metadata.
- No quality gate that blocks push when metadata quality is below threshold.
- No repository-wide consistency linter for sectioning style in all `*_message_examples.txt` files.
- Category confidence thresholds are currently fixed (not configurable by CLI flags).

## Practical Quality Checklist

Before pushing new character data:

1. Ensure source text is structured with clear section boundaries.
2. Run analysis with review output:

```bash
uv run python scripts/rag/analyze_rag_text.py analyze rag_data/<name>.txt \
  -o rag_data/<name>.json \
  --review-report rag_data/<name>_review.json \
  --strict
```

3. Validate metadata:

```bash
uv run python scripts/rag/analyze_rag_text.py validate rag_data/<name>.json
```

4. Push and verify retrieval:

```bash
uv run python scripts/rag/push_rag_data.py rag_data/<name>.txt -c <name> -w
uv run python scripts/rag/manage_collections.py test <name> -q "<sample query>" -k 5
```

## Related Files

- `scripts/rag/analyze_rag_text.py`
- `scripts/rag/push_rag_data.py`
- `scripts/rag/manage_collections.py`
- `tests/test_rag_scripts.py`
