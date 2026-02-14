# Dynamic Context Allocation - Visual Guide

## Context Window Breakdown

### Small Model (4K context - e.g., Mistral 7B-Instruct Q3)

```
Total Context: 4096 tokens
│
├─ Response Buffer: 256 tokens (reserved)
├─ System Prompt: 800 tokens
│  ├─ Character description
│  ├─ Scenario
│  └─ Instructions
│
└─ Available for Content: ~3040 tokens
   ├─ User Input: ~300 tokens (10%)
   ├─ Conversation History: ~912 tokens (30%)
   │  └─ Recent 2-3 turns
   ├─ Message Examples: ~728 tokens (25%)
   │  └─ Character behavior demos
   └─ Vector Context (RAG): ~1100 tokens
      └─ Retrieved documents/context
```

### Large Model (8K context - e.g., Llama 3.1-8B with rope_scale)

```
Total Context: 8192 tokens
│
├─ Response Buffer: 256 tokens (reserved)
├─ System Prompt: 800 tokens (same as above)
│
└─ Available for Content: ~7136 tokens
   ├─ User Input: ~700 tokens (10%)
   ├─ Conversation History: ~2140 tokens (30%)
   │  └─ Most/all of recent conversation
   ├─ Message Examples: ~1785 tokens (25%)
   │  └─ More extensive examples
   └─ Vector Context (RAG): ~2511 tokens
      └─ Much more retrieved context
```

## Allocation Algorithm

```
START: Receive user message
│
├─ Calculate Budget
│  ├─ Read model's n_ctx (context window)
│  ├─ Subtract reserved response space
│  └─ Subtract system prompt size
│  └─ Result: Available tokens for content
│
├─ Collect Raw Content
│  ├─ Retrieve up to 32 RAG chunks (initially)
│  ├─ Prepare message examples
│  └─ Build full conversation history
│
├─ Allocate Dynamically
│  ├─ Reserve user input space (~10% of budget)
│  ├─ Allocate conversation history (~30% of remaining)
│  │  └─ Respect MIN_HISTORY_TURNS and MAX_HISTORY_TURNS
│  ├─ Allocate message examples (~25% of remaining)
│  │  └─ Truncate at paragraph boundaries
│  └─ Allocate RAG context (rest)
│     └─ Truncate at sentence/chunk boundaries
│
├─ Build Final Prompt
│  ├─ System prompt (full)
│  ├─ Allocated message examples
│  ├─ Allocated conversation history
│  ├─ Allocated vector context
│  └─ User message
│
├─ Send to Model
│  └─ Model responds within budget
│
END
```

## Token Flow Example

```
User enters: "What are the security implications of..."

┌─────────────────────────────────────────┐
│ Process User Message                    │
└─────────────────────────┬───────────────┘
                          │
                    ┌─────▼─────┐
                    │ Calculate │
                    │  Budget   │
                    └─────┬─────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
   System Prompt    RAG Retrieval      History Built
   (800 tokens)     (32 chunks)        (all turns)
        │                 │                 │
        │       ┌─────────┴─────────┐       │
        │       │ Token Counter    │       │
        │       └────────┬──────────┘       │
        │                │                  │
        │  ┌─────────────┼──────────────┐   │
        │  │             │              │   │
        ▼  ▼             ▼              ▼   ▼
   ┌─────────────────────────────────────────┐
   │  Dynamic Allocator                      │
   │                                         │
   │  Budget = 3040 tokens                   │
   │  Input (10%):     300 tokens    ✓      │
   │  History (30%):   912 tokens    ✓      │
   │  Examples (25%):  728 tokens    ✓      │
   │  Context (rest): 1100 tokens    ✓      │
   └────────────────┬────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
   [Examples]  [Context]  [History]
   728 tokens  1100 toks  912 tokens
        │           │           │
        └───────────┼───────────┘
                    │
                    ▼
              ┌──────────────┐
              │ Final Prompt │
              ├──────────────┤
              │ System: 800  │
              │ Docs: 1100   │
              │ Examples: 728│
              │ History: 912 │
              │ Input: 300   │
              │ Total: 3840  │
              └──────────────┘
                    │
                    ▼
              ┌──────────────┐
              │ Pass to LLM  │
              │ 4096 budget  │
              │ 256 for resp │
              │ Result: ✓    │
              └──────────────┘
```

## Comparison: Static vs Dynamic

### Static Approach (RAG_K=2)
```
Always retrieve 2 chunks, no matter model size:

Small Model (4K):
├─ Waste context with only 2 chunks
└─ Could use 10+ chunks if space available

Large Model (8K):
├─ Still only 2 chunks
└─ Vastly underutilized capacity
```

### Dynamic Approach
```
Retrieve and allocate based on available budget:

Small Model (4K):
├─ System Prompt: 800
├─ Available: 3040
└─ Context Chunks: ~8-10 chunks (1100 tokens)

Large Model (8K):
├─ System Prompt: 800
├─ Available: 7136
└─ Context Chunks: ~20-25 chunks (2500 tokens)
```

## Allocation Boundary Decisions

```
Context Chunks
│
├─ Chunks 1-5: Definitely include (high relevance)
│
├─ Chunks 6-10: Include if space allows ◄── Dynamic boundary
│
├─ Chunks 11-15: Only with large budget
│
└─ Chunks 16+: Rarely included
    (unless 70B+ model with 8K+ context)

Conversation History
│
├─ Last turn: Always (MIN_HISTORY_TURNS=1)
│
├─ Last 2-3 turns: Usually if space
│
├─ Last 4-8 turns: With large budget ◄── Dynamic boundary
│
└─ Entire conversation: Only very large models
    (and configurable MAX_HISTORY_TURNS limit)

Message Examples
│
├─ Partial: First few examples when budget limited
│
└─ Full: All examples when budget available ◄── Dynamic scaling
```

## Configuration Effects

### Scenario: Different Settings on Same 4K Model

#### Conservative (focused on coherence)
```json
"MIN_HISTORY_TURNS": 3,
"MAX_HISTORY_TURNS": 5,
"RESERVED_FOR_RESPONSE": 512
```
Results in:
- More history (better coherence)
- Less context (fewer RAG chunks)
- Shorter responses possible

#### Aggressive (maximize context)
```json
"MIN_HISTORY_TURNS": 1,
"MAX_HISTORY_TURNS": 3,
"RESERVED_FOR_RESPONSE": 128
```
Results in:
- Less history (only recent turn)
- More context (many RAG chunks)
- Longer responses possible

## Real-World Examples

### Example 1: Customer Support Bot (Shodan Character)

First turn with 4K model:
```
Budget Available: 3040 tokens
├─ System: 900 (character description, rules)
├─ History: 0 (first turn)
├─ Examples: 800 (behavior demonstrations)
└─ Context: 1340 (security knowledge base)

User: "What's the CVE-2024-1234 impact?"
Model: [Retrieves relevant security docs, responds in character]
```

5th turn in same conversation:
```
Budget Available: 3040 tokens
├─ System: 900
├─ History: 900 (last 2-3 exchanges for context)
├─ Examples: 300 (less needed, model has pattern)
└─ Context: 940 (fewer docs but still relevant)

User: "What about mitigation?"
Model: [Has prior turns for context, fewer but focused docs]
```

### Example 2: Creative Writing (Large Model)

With 8K Llama 3.1-8B model:
```
Budget Available: 7136 tokens
├─ System: 800
├─ History: 2500 (keep rich conversation context)
├─ Examples: 2000 (detailed character examples)
└─ Context: 1836 (world-building document)

User: "Continue the story from where we left off..."
Model: [Rich context, multiple prior turns, detailed guidelines]
Result: Highly coherent, long creative response
```

## Monitoring Allocation

Enable debugging to see decisions:

```json
{
  "DEBUG_CONTEXT": true,
  "SHOW_LOGS": true
}
```

Output each turn:
```
Context Window: 4096 tokens
  Response Buffer: 256 tokens
  System Prompt: 842 tokens
  Available: 2998 tokens
Allocation:
  Input: 185 tokens
  History: 896 tokens (3 turns)
  Examples: 722 tokens
  Context: 1195 tokens (7 chunks used)
  Total: 2998 / 2998
```

Track improvements over time:
- Did response quality improve?
- Is history being preserved?
- Are RAG chunks helping?
- Any token overflow warnings?
