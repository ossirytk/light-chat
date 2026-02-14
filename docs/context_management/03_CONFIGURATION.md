# Configuration: Dynamic Context Window

## All Configuration Options

Located in: `configs/appconf.json`

### Quick Reference

```json
{
  "USE_DYNAMIC_CONTEXT": false,
  "DEBUG_CONTEXT": false,
  "RESERVED_FOR_RESPONSE": 256,
  "MIN_HISTORY_TURNS": 1,
  "MAX_HISTORY_TURNS": 8,
  "CHUNK_SIZE_ESTIMATE": 150,
  "MAX_INITIAL_RETRIEVAL": 20
}
```

## Detailed Options

### USE_DYNAMIC_CONTEXT
**Type:** Boolean  
**Default:** `false`  
**Effect:** Enable/disable dynamic context allocation

```json
{
  "USE_DYNAMIC_CONTEXT": false
}
```

| Value | Behavior |
|-------|----------|
| `false` | Static mode (original behavior). No overhead. |
| `true` | Dynamic allocation on turns 2+. Better context utilization. |

**When to use each:**
- `false`: Most users (default, safe, predictable)
- `true`: Testing or models where context budget matters

---

### DEBUG_CONTEXT
**Type:** Boolean  
**Default:** `false`  
**Effect:** Log detailed allocation info per turn

```json
{
  "DEBUG_CONTEXT": true,
  "SHOW_LOGS": true
}
```

When enabled, outputs:
```
Context Window: 4096 tokens
  Response Buffer: 256 tokens
  System Prompt: 842 tokens
  Available: 2998 tokens
Allocation:
  Input: 187 tokens
  History: 896 tokens
  Examples: 721 tokens
  Context: 1194 tokens
  Total: 2998 / 2998
```

**When to enable:**
- Testing the system
- Verifying allocations make sense
- Troubleshooting unexpected behavior

---

### RESERVED_FOR_RESPONSE
**Type:** Integer (tokens)  
**Default:** `256`  
**Range:** 50-1000  
**Effect:** Tokens reserved for model output

```json
{
  "RESERVED_FOR_RESPONSE": 256
}
```

| Value | Result |
|-------|--------|
| 128 | More context budget, shorter responses possible |
| 256 | Balanced (recommended) |
| 512 | Less context budget, more response space |
| 1024 | Minimal context, maximum response length |

**Tune this if:**
- Responses get cut off → increase to 512+
- Model responses too short → increase
- Want more context → decrease to 128-200
- Running out of VRAM → decrease (trade response length)

---

### MIN_HISTORY_TURNS
**Type:** Integer (turns)  
**Default:** `1`  
**Range:** 1-10  
**Effect:** Minimum conversation turns to always preserve

```json
{
  "MIN_HISTORY_TURNS": 1
}
```

| Value | Behavior |
|-------|----------|
| 1 | Keep at least last exchange (coherence baseline) |
| 2 | Keep at least 2 exchanges (better context) |
| 3+ | Keep more history (good for long conversations) |

**Why it matters:**
- Ensures model references recent turns
- Prevents losing conversation context

**Tune this if:**
- Model forgets earlier context → increase to 2-3
- Conversation references go back far → increase to 3+
- Memory constrained → keep at 1

---

### MAX_HISTORY_TURNS
**Type:** Integer (turns)  
**Default:** `8`  
**Range:** 1-20  
**Effect:** Never include more than this many turns

```json
{
  "MAX_HISTORY_TURNS": 8
}
```

| Value | Result |
|-------|--------|
| 1 | Only current + previous (minimal history) |
| 3 | Last few exchanges |
| 8 | Most recent X turns (recommended) |
| 12+ | Near-full conversation (large models only) |

**Tune this for:**
- **Short conversations:** 3-5
- **Balanced:** 8 (default)
- **Creative writing:** 12-20
- **Q&A mode:** 1-2

**Combining with MIN_HISTORY_TURNS:**
```json
{
  "MIN_HISTORY_TURNS": 2,    // At least 2
  "MAX_HISTORY_TURNS": 8     // At most 8
}
```
Result: Dynamic history of 2-8 turns based on budget.

---

### CHUNK_SIZE_ESTIMATE
**Type:** Integer (tokens)  
**Default:** `150`  
**Effect:** Estimated tokens per RAG chunk (used for initial k calculation)

```json
{
  "CHUNK_SIZE_ESTIMATE": 150
}
```

| Value | Use Case |
|-------|----------|
| 100 | Many small chunks |
| 150 | Standard chunks (recommended) |
| 200+ | Large, dense chunks |

**How it's used:**
```python
initial_k = available_budget / chunk_size_estimate
```

If your chunks are actually ~200 tokens but estimate is 150, you'll retrieve fewer chunks than you have budget for (conservative).

**To tune:**
1. Check actual chunk sizes in ChromaDB
2. Count a few chunks manually
3. Set to average size

---

### MAX_INITIAL_RETRIEVAL
**Type:** Integer (chunks)  
**Default:** `20`  
**Range:** 5-100  
**Effect:** Never retrieve more than this many chunks from RAG

```json
{
  "MAX_INITIAL_RETRIEVAL": 20
}
```

| Value | Impact |
|-------|--------|
| 5 | Very conservative, few chunks |
| 10 | Moderate (good for Q&A) |
| 20 | Standard (recommended) |
| 50+ | Aggressive (large models) |

**Why cap it?**
- Prevents expensive ChromaDB queries
- Protects against metric overflow
- Maintains reasonable latency

**Tune this if:**
- Query takes too long → decrease to 10
- Want more chunks → increase to 30-50
- Have large, expensive RAG → decrease to 5-10

---

## Configuration Profiles

### Profile 1: Safe Mode (Default)
```json
{
  "USE_DYNAMIC_CONTEXT": false,
  "DEBUG_CONTEXT": false
}
```

**Use for:** Production, stability first  
**Characteristics:** No overhead, predictable behavior  
**Performance:** Baseline

---

### Profile 2: Testing
```json
{
  "USE_DYNAMIC_CONTEXT": true,
  "DEBUG_CONTEXT": true,
  "SHOW_LOGS": true,
  "MIN_HISTORY_TURNS": 1,
  "MAX_HISTORY_TURNS": 4
}
```

**Use for:** Experimenting, seeing how it works  
**Characteristics:** Verbose logging, limited history  
**Performance:** Minimal impact

---

### Profile 3: Question-Answering
```json
{
  "USE_DYNAMIC_CONTEXT": true,
  "DEBUG_CONTEXT": false,
  "MIN_HISTORY_TURNS": 1,
  "MAX_HISTORY_TURNS": 2,
  "RESERVED_FOR_RESPONSE": 256,
  "MAX_INITIAL_RETRIEVAL": 30
}
```

**Use for:** Q&A, focused answers  
**Characteristics:** Minimal history, maximum context  
**Performance:** More RAG, less overhead

---

### Profile 4: Creative Writing
```json
{
  "USE_DYNAMIC_CONTEXT": true,
  "DEBUG_CONTEXT": false,
  "MIN_HISTORY_TURNS": 3,
  "MAX_HISTORY_TURNS": 12,
  "RESERVED_FOR_RESPONSE": 512,
  "CHUNK_SIZE_ESTIMATE": 200
}
```

**Use for:** Story generation, character consistency  
**Characteristics:** Rich history, more response space  
**Performance:** Balanced

---

### Profile 5: Limited Memory
```json
{
  "USE_DYNAMIC_CONTEXT": false,
  "RESERVED_FOR_RESPONSE": 128,
  "MIN_HISTORY_TURNS": 1,
  "MAX_HISTORY_TURNS": 2
}
```

**Use for:** Low VRAM, tight constraints  
**Characteristics:** Static mode, short history, conservative  
**Performance:** Minimal memory

---

### Profile 6: High-End GPU
```json
{
  "USE_DYNAMIC_CONTEXT": true,
  "DEBUG_CONTEXT": false,
  "MIN_HISTORY_TURNS": 4,
  "MAX_HISTORY_TURNS": 16,
  "RESERVED_FOR_RESPONSE": 512,
  "MAX_INITIAL_RETRIEVAL": 50
}
```

**Use for:** 70B+ models, high VRAM  
**Characteristics:** Aggressive context usage, rich history  
**Performance:** Maximum utilization

---

## Tuning Guide

### Issue: Responses Get Cut Off

**Symptoms:** Model stops mid-sentence

**Solution:**
```json
{
  "RESERVED_FOR_RESPONSE": 512
}
```

**Why:** Model runs out of response space

---

### Issue: History Forgotten Too Quickly

**Symptoms:** Model doesn't reference earlier conversation

**Solution:**
```json
{
  "MIN_HISTORY_TURNS": 3,
  "MAX_HISTORY_TURNS": 12
}
```

**Why:** More history = better coherence

---

### Issue: GPU Spam, Slow Queries

**Symptoms:** Retrieve seems to hang, high GPU load

**Solution:**
```json
{
  "MAX_INITIAL_RETRIEVAL": 10,
  "USE_DYNAMIC_CONTEXT": false
}
```

**Why:** Fewer chunks = faster queries

---

### Issue: Not Enough Context/Knowledge

**Symptoms:** Model doesn't know relevant facts

**Solution:**
```json
{
  "USE_DYNAMIC_CONTEXT": true,
  "MIN_HISTORY_TURNS": 1,
  "MAX_HISTORY_TURNS": 3,
  "MAX_INITIAL_RETRIEVAL": 30
}
```

**Why:** More space for RAG context (less history)

---

### Issue: Conversations Too Long

**Symptoms:** Prompt keeps growing, inference slows down

**Solution:**
```json
{
  "MAX_HISTORY_TURNS": 4
}
```

**Why:** Limit conversation history length

---

## Advanced: Custom Allocation Ratios

Edit `context_manager.py` lines ~170-180:

```python
# Current ratios (in _allocate_content method):
history_budget = remaining_budget * 0.30    # 30% for history
examples_budget = remaining_budget * 0.25   # 25% for examples
context_budget = remaining              # 45% for context

# Example: More context, less history
history_budget = remaining_budget * 0.15    # 15%
examples_budget = remaining_budget * 0.15   # 15%
context_budget = remaining              # 70%
```

**Warning:** Requires code edit, affects all conversations.

---

## Environment Variables

Currently: None  

All configuration via `configs/appconf.json`.

Future: May support `LIGHT_CHAT_*` env vars.

---

## Migration Guide

### From Static to Dynamic

1. **Backup config:**
```bash
cp configs/appconf.json configs/appconf.json.backup
```

2. **Enable dynamic:**
```json
{
  "USE_DYNAMIC_CONTEXT": true,
  "DEBUG_CONTEXT": true
}
```

3. **Test a few turns:**
```bash
python main.py
```

4. **Monitor logs** for allocation info

5. **Adjust if needed:**
- Too many chunks? Lower `MAX_INITIAL_RETRIEVAL`
- History lost? Raise `MIN_HISTORY_TURNS`
- Responses short? Raise `RESERVED_FOR_RESPONSE`

6. **Disable debugging:**
```json
{
  "DEBUG_CONTEXT": false
}
```

---

## Best Practices

1. **Start with defaults** - don't change unless needed
2. **Enable DEBUG_CONTEXT** while tuning - see what's happening
3. **Test one change at a time** - measure impact
4. **Monitor conversation length** - watch for drift
5. **Profile your chunks** - know actual token counts
6. **Keep backups** - save working configs

---

## Next Steps

- **Quick start?** → See `01_QUICKSTART.md`
- **How it works?** → See `02_HOW_IT_WORKS.md`
- **Real examples?** → See `04_EXAMPLES.md`
- **Troubleshooting?** → See `06_TROUBLESHOOTING.md`
