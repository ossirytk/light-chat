# Light Chat

Character-focused local chatbot with RAG support (ChromaDB + LangChain), CLI and web entrypoints, and tooling for metadata generation and collection management.

## Docs Quick Links

- Detailed RAG management docs: `docs/rag_management/00_README.md`
- RAG scripts guide: `docs/RAG_SCRIPTS_GUIDE.md`
- Context management docs: `docs/context_management/00_README.md`
- Config files docs: `docs/configs/00_README.md`
- Future work: `docs/future_work/`

## What It Includes

- Local chat runtime backed by `llama-cpp-python`
- Character-card-driven prompting (`cards/*.json`)
- RAG retrieval from ChromaDB collections
- Dynamic context budgeting and history management
- GPU offload auto-layer calculation and KV cache quant support
- Scripted workflows for analyzing, pushing, and managing RAG data

## Current Runtime Entry Points

- CLI chat: `main.py`
- Web chat (FastAPI + Jinja2 + HTMX): `web_app.py`

Run either with `uv`:

```powershell
uv run python main.py
uv run uvicorn web_app:app --host 127.0.0.1 --port 8000
```

Primary desktop workflow:

- Open the repository from the Windows dev drive in VS Code.
- Use the integrated PowerShell terminal to run the `uv` commands above.
- WSL/Ubuntu with `fish` remains a supported alternative workflow if you still use it.

There is currently no checked-in `.vscode/tasks.json` or `.vscode/launch.json`, so the documented `uv` commands above are the source of truth for starting the app.

If you create local VS Code tasks or debug profiles, point them at the same `uv` commands rather than assuming a Unix-only shell.

Stop the web server from another terminal:

PowerShell:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen |
  Select-Object -ExpandProperty OwningProcess -Unique |
  ForEach-Object { Stop-Process -Id $_ }
```

WSL/Unix alternative:

```bash
pkill -f 'uvicorn web_app:app'
```

Web diagnostics endpoints:

```powershell
curl.exe -s http://127.0.0.1:8000/health
curl.exe -s http://127.0.0.1:8000/healthz/full
curl.exe -s http://127.0.0.1:8000/chat/debug
curl.exe -s http://127.0.0.1:8000/chat/debug/history
curl.exe -s http://127.0.0.1:8000/chat/session/list
```

Notes for web chat behavior:

- Shows status updates (`Ready`, `Sending`, `Thinking`, `Streaming`, `Timed out`).
- Applies a stream timeout and surfaces a `Retry` button on stream failure.
- Supports named session save + explicit session picker load in the sidebar.
- Shows both latest retrieval debug stats and per-turn retrieval trace history.
- Provides quick actions for copy/export and command-equivalent controls (`clear`, `reload`, `help`).

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

5. Evaluate retrieval fixtures with summary metrics:

```bash
uv run python -m scripts.rag.manage_collections evaluate-fixtures --fixture-file tests/fixtures/retrieval_fixtures.json
```

Optional report export:

```bash
uv run python -m scripts.rag.manage_collections evaluate-fixtures \
  --fixture-file tests/fixtures/retrieval_fixtures.json \
  --output-json logs/retrieval_eval.json \
  --output-csv logs/retrieval_eval.csv
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
- Architecture remains CLI-first with shared config usage via `configs/config.v2.json`.

This section intentionally focuses on active behavior and omits historical benchmark/commit snapshot details.

## Current Config Notes

Runtime config is defined in `configs/config.v2.json`.

- Start from `configs/config.v2.example.json`.
- Runtime loads `configs/config.v2.json` directly.

### `configs/config.v2.json` (selected active defaults)

```json
{
  "rag": {"collection": "shodan", "k": 3, "k_mes": 2, "use_mmr": true},
  "context": {"dynamic": {"enabled": true}, "history": {"max_turns": 10}},
  "model": {"type": "mistral", "layers": "auto", "target_vram_usage": 0.8, "kv_cache_quant": "f16", "n_ctx": 32768}
}
```

## Documentation Map

- RAG script usage: `docs/RAG_SCRIPTS_GUIDE.md`
- Detailed RAG management docs: `docs/rag_management/00_README.md`
- Context management docs: `docs/context_management/00_README.md`
- Config files docs: `docs/configs/00_README.md`
- GPU layer auto-tuning: `docs/AUTO_GPU_LAYERS.md`
- Flash attention build helper: `docs/FLASH_ATTENTION_BUILD.md`
- Future work status: `docs/future_work/`

## Testing and Linting

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

Or run targeted tests:

```bash
uv run pytest tests/test_rag_scripts.py
uv run pytest tests/test_response_processing.py
```

## License

See `LICENSE`.
