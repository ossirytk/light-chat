# Quick Start Guide: Dynamic Context Window

## In 30 Seconds

The dynamic context system is **already enabled**. Here's what it does:

1. **Detects** your model's context size (e.g., 4096 tokens)
2. **Calculates** how many tokens are available after system prompt and response reserve
3. **Automatically scales** RAG chunks, message examples, and conversation history to fit
4. **Maintains quality** by respecting minimum coherence requirements

Result: Your chatbot gets the most out of available context, whether you're using a 3.8B or 70B model.

## Before vs After

### Before (Static Configuration)
```python
# Your configuration
self.rag_k = 2  # Always retrieve exactly 2 chunks
self.user_message_history = deque(maxlen=3)  # Always keep 3 turns max

# Problem: Wasted capacity on large models, squeezed on small ones
```

### After (Dynamic Configuration)
```json
{
  "USE_DYNAMIC_CONTEXT": true,  // Enabled by default
  "RESERVED_FOR_RESPONSE": 256,
  "MIN_HISTORY_TURNS": 1,
  "MAX_HISTORY_TURNS": 8
}

# Result: Automatically scales RAG_K and history to fit available budget
```

## Configuration Settings

### Three Levels of Customization

#### Level 1: Just Use It (Recommended)
```json
{
  "USE_DYNAMIC_CONTEXT": true
}
// Everything else uses smart defaults
```

#### Level 2: Tune Response Space
```json
{
  "USE_DYNAMIC_CONTEXT": true,
  "RESERVED_FOR_RESPONSE": 256    // More room for output?
}
```

#### Level 3: Control History Balance
```json
{
  "USE_DYNAMIC_CONTEXT": true,
  "MIN_HISTORY_TURNS": 1,       // Keep at least this
  "MAX_HISTORY_TURNS": 8,       // Never exceed this
  "RESERVED_FOR_RESPONSE": 256
}
```

## What Gets Dynamically Allocated?

| Component | Static | Dynamic |
|-----------|--------|---------|
| **RAG Chunks** | Fixed `RAG_K=2-7` | Scales: 2-25+ based on budget |
| **Examples** | Full or nothing | Scales 0-100% to fit |
| **History** | Fixed `deque(maxlen=3)` | 1-8 turns based on budget |
| **System Prompt** | Full (always) | Full (always) |

## Try It Now

### Option 1: No Changes Required
If you're happy with defaults, just run normally:
```bash
python main.py
```
The system automatically detects your model's context and starts optimizing.

### Option 2: Enable Debugging
See what's happening:
```json
// configs/appconf.json
{
  "USE_DYNAMIC_CONTEXT": true,
  "DEBUG_CONTEXT": true,
  "SHOW_LOGS": true,
  "LOG_LEVEL": "DEBUG"
}
```

Each turn will show:
```
Context Window: 4096 tokens
  System Prompt: 842 tokens
  Available: 3254 tokens
Allocation:
  Input: 187 tokens
  History: 896 tokens
  Examples: 721 tokens
  Context: 1450 tokens
```

### Option 3: Turn it Off (to compare)
```json
{
  "USE_DYNAMIC_CONTEXT": false
}
// Falls back to static RAG_K behavior
```

## Practical Examples

### Small GPU, Large Context Model
```json
{
  "USE_DYNAMIC_CONTEXT": true,
  "RESERVED_FOR_RESPONSE": 128,  // Be conservative with output
  "MIN_HISTORY_TURNS": 1,
  "MAX_HISTORY_TURNS": 4          // Limit history
}
```
Result: More room for RAG context, fewer history turns.

### High-End GPU, Small Context Model
```json
{
  "USE_DYNAMIC_CONTEXT": true,
  "RESERVED_FOR_RESPONSE": 512,   // Generous for detailed responses
  "MIN_HISTORY_TURNS": 2,
  "MAX_HISTORY_TURNS": 12         // Keep rich conversation
}
```
Result: Balanced distribution, good consistency.

### Question-Answering Mode
```json
{
  "USE_DYNAMIC_CONTEXT": true,
  "MIN_HISTORY_TURNS": 1,         // Don't need much history
  "MAX_HISTORY_TURNS": 2,
  "RESERVED_FOR_RESPONSE": 256
}
```
Result: Maximize RAG context, minimal history.

### Creative Writing Mode
```json
{
  "USE_DYNAMIC_CONTEXT": true,
  "MIN_HISTORY_TURNS": 3,         // Need coherence
  "MAX_HISTORY_TURNS": 8,         // Long context
  "RESERVED_FOR_RESPONSE": 512    // Long responses
}
```
Result: Balanced mix of history, examples, and detail context.

## Common Questions

### Q: Will this work with my existing code?
**A:** Yes! It's backward compatible. If you disable it, behaves like before.

### Q: How much slower is it?
**A:** Negligible. Token counting adds 10-50ms per turn, which is tiny compared to LLM inference (usually 1-30+ seconds).

### Q: Can I use my own tokenizer?
**A:** Yes, but the approximate counter is usually good enough. See `DYNAMIC_CONTEXT.md` for integration.

### Q: What if my context window isn't detected?
**A:** Set it manually in config:
```json
{
  "N_CTX": 4096
}
```

### Q: How do I know if it's working?
**A:** Enable debugging to see which chunks/history/examples are included per turn.

### Q: Should I change RAG_K anymore?
**A:** It's ignored when dynamic context is on, but kept for fallback. You can leave it as-is.

## Troubleshooting

### Issue: Same response quality, no difference
**Check:**
```json
{
  "DEBUG_CONTEXT": true
}
```
to see what's being allocated. Is it actually different from static mode?

### Issue: Context gets truncated
**Solution:** Increase `RESERVED_FOR_RESPONSE`:
```json
{
  "RESERVED_FOR_RESPONSE": 512
}
```

### Issue: History disappears too fast
**Solution:** Increase `MIN_HISTORY_TURNS`:
```json
{
  "MIN_HISTORY_TURNS": 2
}
```

### Issue: Not detecting context window
**Check logs for:**
```
"Could not detect context window, using default"
```
Then set manually:
```json
{
  "N_CTX": 4096
}
```

## Performance Tips

1. **First turn slower?** System is analyzing context budget. Subsequent turns are fast.

2. **Response cuts off?** Increase `RESERVED_FOR_RESPONSE` to 512+

3. **Token counting feels wrong?** The approximate counter is conservative intentionally. Use exact counting if needed.

4. **Want more RAG chunks?** Increase `RESERVED_FOR_RESPONSE` (less space for response = more for context).

5. **Want richer history?** Decrease `RESERVED_FOR_RESPONSE` (if you don't need long responses).

## What Changed

### You DON'T Need to Change:
- Your character cards
- Your RAG documents
- Your main.py script
- Your chromadb setup
- Any conversation flow

### What Changed Automatically:
- Amount of RAG context included (scales with budget)
- Amount of message examples included (scales with budget)
- Amount of conversation history (scales with budget, respects min/max)
- Token calculations for context window

### What's Optional:
- Configuration tuning (only if you want specific behavior)
- Exact token counting (if you want precision)
- Debug logging (to see what's happening)

## Next Steps

1. **Run normally** with defaults (no changes needed)
2. **Check logs** if curious what's being allocated
3. **Tune config** only if you notice specific issues
4. **Read** `DYNAMIC_CONTEXT.md` for deep dive

## File Reference

| File | Purpose |
|------|---------|
| `context_manager.py` | Core dynamic allocation engine |
| `conversation_manager.py` | Integration with chatbot |
| `DYNAMIC_CONTEXT.md` | Complete documentation |
| `CONTEXT_VISUALIZATION.md` | Visual explanations |
| `IMPLEMENTATION.md` | Technical details |
| `configs/appconf.json` | Configuration |

## Support

Check docs in this order:
1. **Quick** → This file
2. **Visual** → `CONTEXT_VISUALIZATION.md`
3. **Complete** → `DYNAMIC_CONTEXT.md`
4. **Technical** → `IMPLEMENTATION.md`

---

**TL;DR:** Dynamic context is enabled by default. Your chatbot now automatically uses available token budget efficiently. Change config only if you want specific behavior.
