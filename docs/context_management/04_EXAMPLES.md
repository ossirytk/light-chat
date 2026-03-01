# Context Management Examples

Last verified: 2026-03-01

## Example 1 — Small-talk turn (RAG skipped)

Input:

```text
hi
```

Expected behavior:

- `_should_skip_rag_for_message` returns true.
- Runtime uses minimal vector context placeholder.
- Prompt remains focused on persona + history.

## Example 2 — Lore-specific question

Input:

```text
What happened on Citadel Station in 2072?
```

Expected behavior:

- Metadata match extraction likely succeeds.
- Filtered retrieval is attempted first, then fallback strategies.
- Context chunks are deduped and clipped before prompt injection.

## Example 3 — Short follow-up without entity match

Conversation:

```text
User: Tell me about SHODAN's origins.
AI: ...
User: and then?
```

Expected behavior:

- Follow-up may skip RAG if no key match and short length.
- Recent conversation history carries continuity.

## Example 4 — Debugging dynamic allocation

Set:

```json
{
  "DEBUG_CONTEXT": true,
  "USE_DYNAMIC_CONTEXT": true
}
```

Then run chat and inspect allocation logs for input/history/examples/context token usage.

## Example 5 — Static fallback mode

Set:

```json
{
  "USE_DYNAMIC_CONTEXT": false
}
```

Expected behavior:

- Retrieval still runs,
- dynamic allocator is not used,
- prompt behavior is easier to compare for regressions.
