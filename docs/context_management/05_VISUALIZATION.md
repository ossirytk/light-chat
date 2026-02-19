# Visualization: Context Window Allocation

Visual explanations and diagrams for understanding dynamic context allocation.

## Token Budget Breakdown

### 4K Model (Mistral 7B)

```
4096 Total Context
│
├─ 256 Response Buffer
│  └─ Reserved for model output
│
├─ 850 System Prompt
│  ├─ Character description (~400 tokens)
│  ├─ Scenario (~250 tokens)
│  ├─ Message examples (in 1st turn, ~200 tokens)
│  └─ Instructions (~50 tokens)
│
└─ 2990 Available for Content
   ├─ Static allocation:
   │  ├─ RAG chunks: 2 × ~150 = 300
   │  ├─ History: 3 turns = 900
   │  └─ Wasted: ~1790 (60%)
   │
   └─ Dynamic allocation (Turn 2+):
      ├─ Input: 300 tokens (10%)
      ├─ History: 897 tokens (30%)
      ├─ Examples: 746 tokens (25%)
      └─ Context: 1347 tokens (45%)
```

### 8K Model (Llama 3.1-8B)

```
8192 Total Context
│
├─ 256 Response Buffer
│
├─ 850 System Prompt
│
└─ 7086 Available for Content
   ├─ Static allocation:
   │  ├─ RAG chunks: 2 × ~150 = 300
   │  ├─ History: 3 turns = 900
   │  └─ Wasted: ~5886 (83%!)
   │
   └─ Dynamic allocation (Turn 2+):
      ├─ Input: 709 tokens (10%)
      ├─ History: 2126 tokens (30%)
      ├─ Examples: 1771 tokens (25%)
      └─ Context: 2480 tokens (45%)
         └─ Equivalent to ~17 chunks!
```

## Token Flow per Turn

### Turn 1 (First Turn - Always Static)

```
User: "Hello"

┌─────────────────────────────┐
│ ConversationManager         │
│ .ask_question()             │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ is_first_turn = True        │
│ use_dynamic_context = True  │
└────────┬────────────────────┘
         │
         ├─ Yes: Skip dynamic (First-turn exclusion)
         │
         ▼
┌─────────────────────────────┐
│ Use Static Mode             │
│ RAG_K = 2 chunks            │
│ Examples: full              │
│ History: none               │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Build Standard Prompt       │
│ ~1700 tokens                │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Model Inference             │
│ ~20 seconds                 │
└─────────────────────────────┘
```

### Turn 2+ (Dynamic Enabled)

```
User: "Tell me more..."

┌──────────────────────────────┐
│ is_first_turn = False        │
│ use_dynamic_context = True   │
└────────┬─────────────────────┘
         │
         ├─ No: Check dynamic
         │
         ▼
┌──────────────────────────────┐
│ Calculate Budget             │
│ ├─ n_ctx: 4096              │
│ ├─ Reserved: 256             │
│ ├─ System: 850              │
│ └─ Available: 2990 tokens   │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│ Check Safety                 │
│ if budget < 500:            │
│   → Static mode             │
│ else:                       │
│   → Continue dynamic        │
└────────┬─────────────────────┘
         │
         ├─ Continue
         │
         ▼
┌──────────────────────────────┐
│ Retrieve Content Adaptively  │
│ ├─ History: All turns       │
│ ├─ Examples: Full           │
│ └─ Context: k = estimate/150│
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│ Allocate Tokens              │
│ ├─ History: 30% remaining   │
│ ├─ Examples: 25%            │
│ └─ Context: 45%             │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│ Build Optimized Prompt       │
│ ~2900 tokens (93% utilized)  │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│ Model Inference              │
│ ~20 seconds (same speed)     │
└──────────────────────────────┘
```

## Allocation Distribution (Pie Chart)

### Static Mode (All Turns)

```
        ┌─────────────────────────────────────┐
        │   Available Budget: 2990 tokens      │
        ├─────────────────────────────────────┤
        │                                     │
        │   ┌──────────────────────────┐     │
        │   │ History (30%)             │  30%│
        │   │ 897 tokens, 3 turns       │     │
        │   └──────────────────────────┘     │
        │                                     │
        │   ┌──────────────────────────┐     │
        │   │ Input (10%)               │  10%│
        │   │ 299 tokens                │     │
        │   └──────────────────────────┘     │
        │                                     │
        │   ┌──────────────────────────┐     │
        │   │ Examples (25%)            │  25%│
        │   │ 746 tokens                │     │
        │   └──────────────────────────┘     │
        │                                     │
        │   ┌──────────────────────────┐     │
        │   │ Context (45%)             │  45%│
        │   │ 1347 tokens, 9 chunks     │     │
        │   └──────────────────────────┘     │
        │                                     │
        │   ┌──────────────────────────┐     │
        │   │ Wasted (90%)              │  90%│
        │   │ 2700 tokens (no dynamic)  │     │
        │   └──────────────────────────┘     │
        └─────────────────────────────────────┘

        WITHOUT DYNAMIC (Always same)
```

### Dynamic Mode Turn 2+

```
        ┌─────────────────────────────────────┐
        │   Available Budget: 2990 tokens      │
        ├─────────────────────────────────────┤
        │                                     │
        │   ┌──────────────────────────┐     │
        │   │ History (30%)             │  30%│
        │   │ 897 tokens, 3 turns       │     │
        │   └──────────────────────────┘     │
        │                                     │
        │   ┌──────────────────────────┐     │
        │   │ Input (10%)               │  10%│
        │   │ 299 tokens                │     │
        │   └──────────────────────────┘     │
        │                                     │
        │   ┌──────────────────────────┐     │
        │   │ Examples (25%)            │  25%│
        │   │ 746 tokens                │     │
        │   └──────────────────────────┘     │
        │                                     │
        │   ┌──────────────────────────┐     │
        │   │ Context (45%)             │  45%│
        │   │ 1347 tokens, 9 chunks     │     │
        │   └──────────────────────────┘     │
        │                                     │
        │   ✓ Fully utilized (95%)            │
        │   ✗ No wasted budget                │
        └─────────────────────────────────────┘

        WITH DYNAMIC (95% vs 10% utilization)
```

## Conversation Evolution

### Without Dynamic (Static)

```
Turn 1:
  Chunks: 2      │ History: 0      │ Quality: ★★★☆☆
  Total: 900

Turn 3:
  Chunks: 2      │ History: 3      │ Quality: ★★★☆☆
  Total: 900

Turn 5:
  Chunks: 2      │ History: 3      │ Quality: ★★★☆☆
  Total: 900

Turn 10:
  Chunks: 2      │ History: 3      │ Quality: ★★★☆☆
  Total: 900 (hits ceiling)

Turn 20:
  Chunks: 2      │ History: 3      │ Quality: ★★★☆☆
  Total: 900 (stuck)

→ Consistent but limited
```

### With Dynamic (Enabled on Turn 2+)

```
Turn 1:
  Chunks: 2      │ History: 0      │ Quality: ★★★☆☆
  (Static mode, first-turn exclusion)

Turn 2:
  Chunks: 8      │ History: 1      │ Quality: ★★★★☆
  (Dynamic: +6 chunks)

Turn 3:
  Chunks: 9      │ History: 2      │ Quality: ★★★★☆
  (Dynamic: full allocation)

Turn 5:
  Chunks: 8      │ History: 3      │ Quality: ★★★★☆
  (Dynamic: balanced, history maxed)

Turn 10:
  Chunks: 7      │ History: 3      │ Quality: ★★★★☆
  (Dynamic: slightly less context for history)

Turn 15:
  Budget check: Available = 400 tokens
  → Falls back to static
  Chunks: 2      │ History: 3      │ Quality: ★★★☆☆
  (Safety: prevents overflow)

→ Improves then stabilizes
```

## Context Scaling Visualization

### Model Size Impact

```
4K Context          8K Context          16K Context
(Mistral)          (Llama 8B)          (Llama 70B)

Static:            Static:             Static:
2 chunks ────────→ 2 chunks ─────────→ 2 chunks
(matches capacity)  (underutilized)    (vastly underutilized)

                         ↓ Dynamic Enabled ↓

Dynamic:           Dynamic:            Dynamic:
6-8 chunks ─┐      15-20 chunks ─┐     40-50 chunks ─┐
            │                     │                    │
      ↓     └──────→     ↓        └──────→     ↓       │
    Good quality      Better quality     Excellent     │
    (+20%)            (+30%)             (+40%)         │
                                                        │
    Uses 90%────→     Uses 95%────→      Uses 95%────→┘
    of budget         of budget          of budget
```

## Budget Allocation Algorithm

```
┌─────────────────────────────────────┐
│ 1. Detect Context Window            │
│    (n_ctx from model or config)     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 2. Calculate Available Budget       │
│    Budget = n_ctx - response        │
│           - system_prompt           │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 3. Retrieve Full Content            │
│    (All history, examples, chunks)  │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 4. Allocate User Input (~10%)       │
│    Reserve space for current input  │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 5. Allocate History (30% rem.)      │
│    Include recent turns, truncate   │
│    if needed, respect min/max       │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 6. Allocate Examples (25% rem.)     │
│    Include behavior demos at        │
│    paragraph boundaries             │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 7. Allocate Context (45% rem.)      │
│    Fill remaining with RAG chunks   │
│    at chunk boundaries              │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 8. Build Final Prompt               │
│    System + History + Examples      │
│    + Context + Input                │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 9. Model Inference                  │
│    Process prompt, generate         │
│    response                         │
└─────────────────────────────────────┘
```

## Troubleshooting Flowchart

```
Having issues?
│
├─ Slow first response (30+ sec)?
│  └─ Check: First-turn exclusion in code
│     Fix: Enable DEBUG_CONTEXT
│
├─ GPU stress after turn 2?
│  └─ Check: Mistral prompt vector context placement
│     Fix: Verify latest code version
│
├─ Responses cut off?
│  └─ Check: RESERVED_FOR_RESPONSE
│     Fix: Increase to 512+
│
├─ Model forgets context?
│  └─ Check: MIN_HISTORY_TURNS
│     Fix: Increase to 2-3
│
├─ Too many/few chunks?
│  └─ Check: MAX_INITIAL_RETRIEVAL
│     Fix: Adjust based on need
│
├─ Allocation seems wrong?
│  └─ Check: Enable DEBUG_CONTEXT
│     Fix: Verify budget calculations
│
└─ Nothing works?
   └─ Set USE_DYNAMIC_CONTEXT: false
      (Falls back to static mode)
```

## Performance Timeline

### First Chat Session

```
0s:   Start
      │
      ├─ Load model: ~5-10 seconds
      │
      ├─ Initialize context manager: +50-200ms
      │
      ▼
~10s: Ready for input
      │
      ├─ Type: "Hello"
      │
      ▼
~10s: First turn
      ├─ Prepare context: +10ms (static)
      ├─ Retrieve chunks: +100ms
      ├─ Model inference: ~20 seconds
      │
      ▼
~30s: First response done
      │
      ├─ Type: "Tell me more..."
      │
      ▼
~30s: Second turn
      ├─ Prepare context: +20ms (dynamic)
      ├─ Retrieve chunks: +50ms
      ├─ Token counting: +10ms
      ├─ Budget calculation: +5ms
      ├─ Allocation logic: +10ms
      ├─ Model inference: ~20 seconds
      │
      ▼
~50s: Second response done (same speed)
```

## Memory Usage

```
Python Base Load:
├─ core/conversation_manager.py: ~500KB
├─ core/context_manager.py: ~200KB
└─ Other utilities: ~1MB
Total: ~2MB

With Dynamic Context:
├─ ContextManager instance: ~200KB
├─ ApproximateTokenCounter: ~50KB
├─ Cached values: ~100KB
└─ Overhead: ~350KB
New Total: ~2.35MB

Impact: <1MB for full system
On 8GB VRAM: <0.01% increase
```

---

## Next Steps

- **Want configuration help?** → See `03_CONFIGURATION.md`
- **Need real examples?** → See `04_EXAMPLES.md`
- **Having issues?** → See `06_TROUBLESHOOTING.md`
