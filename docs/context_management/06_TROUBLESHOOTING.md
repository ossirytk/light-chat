# Context Management Troubleshooting

Last verified: 2026-03-01

## 1) Dynamic mode seems inactive

Check:

- `USE_DYNAMIC_CONTEXT` is true in `configs/appconf.json`.
- You are not only testing first-turn prompts (dynamic allocation is mainly used after first turn).
- `DEBUG_CONTEXT` is enabled to inspect allocation logs.

## 2) Retrieval returns irrelevant chunks

Check:

- metadata quality in `rag_data/<collection>.json`,
- `RAG_COLLECTION` points to the expected collection,
- `RAG_K`, `RAG_FETCH_K`, and `LAMBDA_MULT` are tuned for your data,
- `RAG_SCORE_THRESHOLD` is set only if using similarity mode (`USE_MMR=false`).

## 3) Responses are very short or fallback text appears

Possible causes:

- stream hit `MAX_STREAM_CHARS`,
- stream remained silent until `MAX_SILENT_STREAM_CHARS`,
- backend streaming error triggered `EMPTY_STREAM_FALLBACK`.

Actions:

- raise `MAX_STREAM_CHARS`,
- inspect logs in `logs/light-chat.log`,
- verify model runtime health.

## 4) History quality degrades over time

Check:

- quality gating warnings in logs,
- `MAX_HISTORY_TURNS` value,
- whether many fallback responses are being written.

## 5) Prompt too large / unstable outputs

Actions:

- reduce `MAX_VECTOR_CONTEXT_CHARS`,
- reduce `RAG_K` and `RAG_K_MES`,
- lower `MAX_INITIAL_RETRIEVAL`,
- verify context window (`N_CTX`, model capabilities, and auto-adjust behavior).

## 6) Small-talk answer sounds over-retrieved

Tune:

- `SMALL_TALK_MAX_WORDS`,
- `FOLLOWUP_RAG_MAX_WORDS`.

## Useful Commands

```bash
uv run python main.py
uv run python chat_tui.py
uv run python scripts/rag/manage_collections.py test shodan -q "SHODAN origin" -k 5
uv run python scripts/rag/analyze_rag_text.py validate rag_data/shodan.json
```
