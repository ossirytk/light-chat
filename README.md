# Light Chat

Character-focused local chatbot with RAG support (ChromaDB + LangChain), CLI and Textual TUI entrypoints, and tooling for metadata generation and collection management.

## What It Includes

- Local chat runtime backed by `llama-cpp-python`
- Character-card-driven prompting (`cards/*.json`)
- RAG retrieval from ChromaDB collections
- Dynamic context budgeting and history management
- GPU offload auto-layer calculation and KV cache quant support
- Scripted workflows for analyzing, pushing, and managing RAG data

## Current Runtime Entry Points

- CLI chat: `main.py`
- Textual TUI chat: `chat_tui.py`
- Web chat (FastAPI + Jinja2 + HTMX): `web_app.py`

Run either with `uv`:

```bash
uv run python main.py
uv run python chat_tui.py
uv run uvicorn web_app:app --host 127.0.0.1 --port 8000
```

Stop the web server (from another terminal):

```bash
pkill -f 'uvicorn web_app:app'
```

Web diagnostics endpoints:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/healthz/full
```

Notes for web chat behavior:

- Shows status updates (`Ready`, `Sending`, `Thinking`, `Streaming`, `Timed out`).
- Applies a stream timeout and surfaces a `Retry` button on stream failure.

## Setup

```bash
uv sync
```

Python requirement is defined in `pyproject.toml` (`>=3.13`).

## Quick RAG Workflow

1. Analyze source text and generate metadata:

```bash
uv run python scripts/rag/analyze_rag_text.py analyze rag_data/shodan.txt -o rag_data/shodan.json --strict
```

2. Validate metadata:

```bash
uv run python scripts/rag/analyze_rag_text.py validate rag_data/shodan.json
```

3. Push text into a collection:

```bash
uv run python scripts/rag/push_rag_data.py rag_data/shodan.txt -c shodan -w
```

4. Test retrieval quality:

```bash
uv run python scripts/rag/manage_collections.py test shodan -q "SHODAN origin" -k 5
```

## Script Surface (Current)

### `scripts/rag/analyze_rag_text.py`

Commands:

- `analyze`
- `validate`
- `scan`

Notable options:

- `--auto-categories/--no-auto-categories`
- `--auto-aliases/--no-auto-aliases`
- `--max-aliases`
- `--strict`
- `--review-report`

### `scripts/rag/push_rag_data.py`

Notable options:

- `-c/--collection-name` (required)
- `-w/--overwrite`
- `-d/--dry-run`
- `-m/--metadata-file`
- `-cs/--chunk-size`
- `-co/--chunk-overlap`
- `-t/--threads`

Notes:

- Leading HTML header comments are stripped before chunking.
- Metadata auto-detection maps `<name>.txt` and `<name>_message_examples.txt` to `<name>.json`.

### `scripts/rag/manage_collections.py`

Commands:

- `list-collections`
- `delete`
- `delete-multiple`
- `test`
- `export`
- `info`

### Compatibility wrappers

Top-level wrappers exist for moved scripts:

- `scripts/analyze_rag_text.py`
- `scripts/push_rag_data.py`
- `scripts/manage_collections.py`
- `scripts/fetch_character_context.py`
- `scripts/build_flash_attention.py`

## Implementation Highlights (verified)

The following implementation points are reflected in current docs and code:

- RAG management script set is in place (`analyze_rag_text`, `push_rag_data`, `manage_collections`).
- Metadata analysis + validation workflow is implemented and tested in `tests/test_rag_scripts.py`.
- Collection management supports listing, deletion, pattern deletion, testing, export, and info commands.
- Script workflows are documented in `docs/RAG_SCRIPTS_GUIDE.md`.
- Architecture remains CLI-first with shared config usage via `configs/appconf.json`.

This section intentionally focuses on active behavior and omits historical benchmark/commit snapshot details.

## Current Config Notes

### `configs/appconf.json` (selected active defaults)

```json
{
  "RAG_COLLECTION": "shodan",
  "RAG_K": 3,
  "RAG_K_MES": 2,
  "USE_MMR": true,
  "USE_DYNAMIC_CONTEXT": true,
  "MAX_HISTORY_TURNS": 10
}
```

### `configs/modelconf.json` (selected active defaults)

```json
{
  "MODEL_TYPE": "mistral",
  "LAYERS": "auto",
  "TARGET_VRAM_USAGE": 0.8,
  "KV_CACHE_QUANT": "f16",
  "N_CTX": 32768
}
```

## Documentation Map

- RAG script usage: `docs/RAG_SCRIPTS_GUIDE.md`
- Context management docs: `docs/context_management/00_README.md`
- GPU layer auto-tuning: `docs/AUTO_GPU_LAYERS.md`
- Flash attention build helper: `docs/FLASH_ATTENTION_BUILD.md`
- Future work status: `docs/future_work/`
- Legacy archived notes: `docs/legacy/`

## Testing and Linting

```bash
uv run ruff format .
uv run ruff check .
uv run python -m unittest
```

Or run targeted tests:

```bash
uv run python -m unittest tests.test_rag_scripts
uv run python -m unittest tests.test_response_processing
```

## License

See `LICENSE`.
