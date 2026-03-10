# Conversation Manager Holistic Review

Date: 2026-03-09
Scope: `core/conversation_manager.py` and supporting integration with `core/context_manager.py`, config, and tests.

## Summary

Conversation-quality and RAG features are individually strong, but several integration seams prevent fully holistic behavior in multi-turn practice.

## Findings

### 1. High: Dynamic context budgeting is inconsistent with prompt construction

- In dynamic mode, allocation computes history/context/examples but only uses allocated context/examples.
- Allocated history is not applied to final prompt assembly.
- Prompt construction still injects full history.

Evidence:

- `core/conversation_manager.py:1304`
- `core/conversation_manager.py:1311`
- `core/conversation_manager.py:1221`
- `core/conversation_manager.py:1245`
- `core/conversation_manager.py:1369`

Impact:

- History consumes full prompt space while retrieval context is pruned as if history were budgeted.
- Retrieval and history controls work against each other instead of composing.

### 2. High: Conversation turn splitting logic in ContextManager is incorrect

- `_split_conversation_turns` appends a turn when `current_turn` starts with `User:` and then resets to the current line.
- This can produce malformed turn segmentation and distort history token budgeting.

Evidence:

- `core/context_manager.py:338`
- `core/context_manager.py:340`

Impact:

- History allocation decisions are based on unstable turn boundaries.

### 3. High: Streamed output can diverge from stored history after quality gating

- Raw content is emitted to user during stream.
- Post-processing/quality-gating runs afterward and may replace history entry with fallback.

Evidence:

- `core/conversation_manager.py:1466`
- `core/conversation_manager.py:1609`
- `core/conversation_manager.py:1614`

Impact:

- User-visible answer and stored history can differ, reducing coherence in subsequent turns.

### 4. Medium: User-turn leakage can still appear in the emitted stream chunk

- User-turn pattern detection occurs before break, but that same chunk is emitted first.

Evidence:

- `core/conversation_manager.py:1443`
- `core/conversation_manager.py:1466`
- `core/conversation_manager.py:1468`

Impact:

- Guard stops further generation but does not reliably prevent visible leakage in the currently processed chunk.

### 5. Medium: Message-example retrieval work is done but dropped in dynamic non-first-turn path

- `_get_vector_context` always retrieves `_mes` chunks.
- Dynamic path ignores `_mes_from_rag_full` and allocates examples from an empty input on non-first-turn.

Evidence:

- `core/conversation_manager.py:1103`
- `core/conversation_manager.py:1302`
- `core/conversation_manager.py:1306`
- `core/conversation_manager.py:1278`

Impact:

- Extra retrieval cost with little or no effect in dynamic non-first-turn flow.

### 6. Medium: Dynamic-context exception fallback does not execute static retrieval

- Exception path logs static fallback but sets empty vector context instead of calling static retrieval.

Evidence:

- `core/conversation_manager.py:1340`
- `core/conversation_manager.py:1342`

Impact:

- Transient dynamic allocation failures can silently disable RAG for the turn.

## Test Coverage Gaps

- No direct regression tests found for:
  - `_split_conversation_turns`
  - Dynamic allocation integration where allocated history is reflected in final prompt

## Recommended Fix Order

1. Align dynamic allocation outputs with final prompt assembly (history/context/examples).
2. Fix `_split_conversation_turns` and add direct tests.
3. Make stream/display behavior consistent with stored history when quality-gating fails.
4. Remove redundant `_mes` retrieval work in dynamic non-first-turn flow or wire it through.
5. Make dynamic exception path perform true static retrieval fallback.
