# How Dynamic Context Works

Last verified: 2026-03-07

## Flow Summary

1. User message arrives in `ConversationManager.ask_question`.
2. System decides whether to skip RAG for the turn:
   - short small-talk (`SMALL_TALK_MAX_WORDS`),
   - short follow-up without metadata matches (`FOLLOWUP_RAG_MAX_WORDS`).
3. If retrieval is used, runtime fetches:
   - lore chunks from `RAG_COLLECTION`,
   - style chunks from `<RAG_COLLECTION>_mes`.
4. Chunks are filtered and deduped (including cross-collection dedupe between lore and `_mes`).
5. Dynamic mode (non-first turn) calculates token budget and allocates content.
6. Older history can be compacted into deterministic summary entries when summarization thresholds are met.
7. Prompt is assembled and streamed.

## Budgeting Logic (`ContextManager`)

Budget is based on:

- model context window (`n_ctx` when detectable),
- reserved output tokens (`RESERVED_FOR_RESPONSE`),
- system prompt token estimate.

Dynamic content allocation priority is:

1. current input,
2. conversation history,
3. message examples,
4. vector context.

## Retrieval Strategy

- Query is enriched with character name.
- Metadata filters are built from key matches.
- Runtime uses fallback filter attempts (`$and` -> `$or` -> unfiltered).
- Retrieval method:
  - MMR when `USE_MMR=true`,
  - similarity search otherwise.
- Optional reranking can reorder top candidates (`rag.rerank.enabled`).
- Optional telemetry logs retrieval details (`rag.telemetry.enabled`).

## Context Cleanup

Before prompt injection, context passes through:

- low-quality chunk filtering,
- near-duplicate signature filtering,
- markdown section dedupe,
- max-char clipping (`MAX_VECTOR_CONTEXT_CHARS`).

## Prompt Build Paths

- `MODEL_TYPE == mistral`: custom prompt string builder.
- Other model types: prompt template path (`configs/conversation_template.json`).

## Post-Generation Safeguards

- Early stream stop on generated user-turn patterns.
- Response cleanup and quality gate before history append.
- History update may compact old turns into summary entries with topic-shift notes.
