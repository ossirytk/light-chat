# RAG Scripts Guide

Last verified: 2026-03-07

This guide documents the current CLI behavior for scripts in `scripts/rag/`.

## Docs Quick Links

- RAG management docs hub: `docs/rag_management/00_README.md`
- RAG scripts guide (this file): `docs/RAG_SCRIPTS_GUIDE.md`
- Context management docs: `docs/context_management/00_README.md`
- Future work docs: `docs/future_work/`

For detailed per-script documentation, see:

- `docs/rag_management/00_README.md`

## Scripts

1. `scripts/rag/analyze_rag_text.py`
2. `scripts/rag/push_rag_data.py`
3. `scripts/rag/manage_collections.py`
4. `scripts/rag/old_prepare_rag.py` (legacy batch helper)
5. `scripts/context/fetch_character_context.py`

Top-level wrappers in `scripts/*.py` are kept for compatibility.

---

## 1) Analyze Text and Metadata

### Analyze

```bash
uv run python scripts/rag/analyze_rag_text.py analyze rag_data/shodan.txt -v
```

Common options:

- `-o, --output`
- `-f, --min-freq`
- `--auto-categories/--no-auto-categories`
- `--auto-aliases/--no-auto-aliases`
- `--max-aliases`
- `--strict`
- `--review-report`

### Validate metadata

```bash
uv run python scripts/rag/analyze_rag_text.py validate rag_data/shodan.json
```

### Scan directory

```bash
uv run python scripts/rag/analyze_rag_text.py scan rag_data/ --auto-generate
```

---

## 2) Push RAG Data to ChromaDB

```bash
uv run python scripts/rag/push_rag_data.py rag_data/shodan.txt -c shodan
```

Common options:

- `-w, --overwrite`
- `-d, --dry-run`
- `-m, --metadata-file`
- `-cs, --chunk-size`
- `-co, --chunk-overlap`
- `-t, --threads`
- `-p, --persist-directory`
- `-k, --key-storage`

Notes:

- Leading HTML header comments are stripped before chunking.
- Metadata file auto-detection maps `<name>.txt` (and `<name>_message_examples.txt`) to `<name>.json`.
- Metadata enrichment workers use `ProcessPoolExecutor` with `spawn` context to avoid Python 3.13 `fork()` deprecation warnings in multithreaded runs.

---

## 3) Manage Collections

### List

```bash
uv run python scripts/rag/manage_collections.py list-collections -v
```

### Delete one

```bash
uv run python scripts/rag/manage_collections.py delete shodan_old -y
```

### Delete multiple

```bash
uv run python scripts/rag/manage_collections.py delete-multiple --pattern "test_*" -y
```

### Test retrieval

```bash
uv run python scripts/rag/manage_collections.py test shodan -q "SHODAN origin" -k 5
```

### Export

```bash
uv run python scripts/rag/manage_collections.py export shodan -o backups/shodan.json
```

### Info

```bash
uv run python scripts/rag/manage_collections.py info shodan
```

### Evaluate fixtures

```bash
uv run python -m scripts.rag.manage_collections evaluate-fixtures --fixture-file tests/fixtures/retrieval_fixtures.json
```

Options:

- `--show-failures`
- `--output-json <path>`
- `--output-csv <path>`
- `--history-csv <path>` (append one summary row per run for trend tracking)

---

## 4) Fetch and Clean Character Context From Web

```bash
uv run python scripts/context/fetch_character_context.py "https://en.wikipedia.org/wiki/Leonardo_da_Vinci" -o rag_data/leonardo_da_vinci.txt
```

Features:

- URL validation (rejects private/internal addresses).
- HTML cleanup and noise filtering.
- Citation marker removal.
- Unicode cleanup and whitespace normalization.

---

## Typical Workflow

```bash
uv run python scripts/rag/analyze_rag_text.py analyze rag_data/new_char.txt -o rag_data/new_char.json --strict
uv run python scripts/rag/analyze_rag_text.py validate rag_data/new_char.json
uv run python scripts/rag/push_rag_data.py rag_data/new_char.txt -c new_char -w
uv run python scripts/rag/manage_collections.py test new_char -q "intro prompt" -k 5
```

## Related Files

- `core/conversation_manager.py`
- `scripts/rag/manage_collections.py`
- `tests/test_rag_scripts.py`
