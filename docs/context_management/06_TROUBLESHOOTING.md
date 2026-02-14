# Troubleshooting: Dynamic Context Window

Common issues and solutions. Start by finding your issue below.

## Issue: Slow First Response (> 30 seconds)

### Symptoms
- First turn takes 30+ seconds
- GPU under load
- Prompt seems large

### Root Cause
Vector context retrieved for first turn (fixed in latest version).

### Solution

**Fix 1: Verify you have latest code**
```bash
git pull  # if using git
```

Check `conversation_manager.py` - should have "First turn exclusion" logic:
```python
if self.use_dynamic_context and not is_first_turn:
    # Only applies turn 2+
```

**Fix 2: Disable dynamic context**
```json
{
  "USE_DYNAMIC_CONTEXT": false
}
```

**Fix 3: Lower MAX_INITIAL_RETRIEVAL**
```json
{
  "MAX_INITIAL_RETRIEVAL": 5
}
```

### Verify
First turn should be ~20 seconds. Second turn may be slightly longer.

---

## Issue: GPU Stress / Out of Memory after 2-3 Turns

### Symptoms
- GPU suddenly maxes out
- OOM errors
- Model hangs

### Root Cause
Mistral prompt was repeating vector context across all blocks (fixed in latest version).

### Solution

**Fix 1: Verify you have latest code**
Check `_build_mistral_prompt()` - vector context should only be in final block:
```python
if vector_context and vector_context.strip() != " ":
    current_content = f"{vector_context}\n\n{current_content}"
```

**Fix 2: Disable dynamic context**
```json
{
  "USE_DYNAMIC_CONTEXT": false
}
```

**Fix 3: Reduce history depth**
```json
{
  "MAX_HISTORY_TURNS": 2,
  "RESERVED_FOR_RESPONSE": 128
}
```

### Verify
Turn 2-3 should be same inference time as turn 1 (~20 seconds).

---

## Issue: Model Stops Responding After Turn X

### Symptoms
- Works fine for first few turns
- Then queries start timing out
- Model hangs or no response

### Root Cause
Token budget exceeded, prompt too large.

### Solution

**Check current config:**
```bash
grep -E "USE_DYNAMIC_CONTEXT|MAX_HISTORY_TURNS" configs/appconf.json
```

**If dynamic enabled:**

Reduce history:
```json
{
  "MAX_HISTORY_TURNS": 3
}
```

Reduce response buffer (allows more context space):
```json
{
  "RESERVED_FOR_RESPONSE": 128
}
```

Increase chunk size estimate (retrieve fewer chunks):
```json
{
  "CHUNK_SIZE_ESTIMATE": 200
}
```

**If still broken:**
```json
{
  "USE_DYNAMIC_CONTEXT": false
}
```

### Verify
Enable debug logging:
```json
{
  "DEBUG_CONTEXT": true,
  "SHOW_LOGS": true
}
```

Watch for: "Using static fallback" or "too small for dynamic allocation"

---

## Issue: Model Doesn't Remember Earlier Conversation

### Symptoms
- References to turn 3 are forgotten by turn 6
- Model can't follow context
- Inconsistent responses

### Root Cause
History getting truncated too aggressively.

### Solution

**Increase minimum history:**
```json
{
  "MIN_HISTORY_TURNS": 2
}
```

**Increase maximum history:**
```json
{
  "MAX_HISTORY_TURNS": 8
}
```

**Disable response buffer reduction:**
```json
{
  "RESERVED_FOR_RESPONSE": 256
}
```

### Verify
Enable debugging:
```json
{
  "DEBUG_CONTEXT": true
}
```

Check logs for: `History: X tokens (Y turns)`

Should show 3-5+ turns being included.

---

## Issue: Responses Cut Off Mid-Sentence

### Symptoms
- "The answer is: " then stops
- Incomplete sentences
- Truncated output

### Root Cause
Insufficient response buffer.

### Solution

**Increase response buffer:**
```json
{
  "RESERVED_FOR_RESPONSE": 512
}
```

This reserves 512 tokens for output instead of 256.

**If using dynamic context, reduce history:**
```json
{
  "MAX_HISTORY_TURNS": 3
}
```

Frees up space for response.

### Verify
Turn 2+: Check if allocations show reduced context:
```
Allocation:
  History: 600 tokens (less)
  Context: 1300 tokens (more)
```

---

## Issue: Very Slow RAG Queries

### Symptoms
- Retrieve seems to hang (10+ seconds)
- Only happens with dynamic enabled
- Static mode is fast

### Root Cause
Retrieving too many chunks (high initial_k).

### Solution

**Lower max retrieval:**
```json
{
  "MAX_INITIAL_RETRIEVAL": 10
}
```

Or more aggressively:
```json
{
  "MAX_INITIAL_RETRIEVAL": 5
}
```

**Increase chunk size estimate:**
```json
{
  "CHUNK_SIZE_ESTIMATE": 200
}
```

Causes fewer initial chunks to be retrieved.

**Disable dynamic context:**
```json
{
  "USE_DYNAMIC_CONTEXT": false
}
```

### Verify
Enable debug logging:
```json
{
  "DEBUG_CONTEXT": true
}
```

Check logs for: `_get_vector_context(message, k=X)`

If X > 20, that's too high.

---

## Issue: "Using static fallback" Message Appears

### Symptoms
- Logs show: "Available budget (X tokens) too small..."
- Happens after several turns
- Falls back to static mode

### Root Cause
Token budget dropped below 500 (safety threshold).

### Solution

**This is OK** - safety feature working as intended.

**To avoid it:**
- Reduce history depth
- Increase response buffer reduction (more context space)
- Disable dynamic context after using initial boost

**To allow it:**
Nothing needed. System handles gracefully.

### Verify
Logs show fallback:
```
[WARNING] Available budget (480 tokens) too small...
Using static fallback
```

This is normal for long conversations.

---

## Issue: "Could Not Detect Context Window" Warning

### Symptoms
- Log shows: "Could not detect context window, using default: 4096"
- System works but might not be optimal

### Root Cause
Model's context window not detected from llama-cpp-python.

### Solution

**Set manually in config:**
```json
{
  "N_CTX": 4096
}
```

Or for larger models:
```json
{
  "N_CTX": 8192
}
```

### How to find your model's actual context:
```bash
# In python
from llama_cpp import Llama
model = Llama(model_path="path/to/model.gguf")
print(model.n_ctx())  # Your actual context window
```

### Verify
After setting N_CTX:
```json
{
  "N_CTX": 4096,
  "USE_DYNAMIC_CONTEXT": true
}
```

Logs should show: "Detected model context window: 4096 tokens"

---

## Issue: Inconsistent Allocation (Different Each Turn)

### Symptoms
- Turn 2: 8 chunks
- Turn 3: 3 chunks
- Turn 4: 12 chunks
- Seems random

### Root Cause
Budget changing based on conversation history growth.

### Solution

This is **expected behavior**.

As history grows:
- Less budget for context
- Fewer chunks retrieved

This is the whole point of dynamic allocation.

**To stabilize:**
```json
{
  "MIN_HISTORY_TURNS": 1,
  "MAX_HISTORY_TURNS": 1
}
```

Keeps history minimal, contexts stable.

Or:
```json
{
  "USE_DYNAMIC_CONTEXT": false
}
```

Static mode: always 2 chunks.

---

## Issue: Token Counting Seems Wrong

### Symptoms
- "3000 tokens available" but prompt only contains 2000
- Allocation numbers don't match actual output
- Token estimates seem off

### Root Cause
Approximate token counter is conservative (by design).

### Solution

This is **expected behavior**. The counter estimates conservatively to prevent overflow.

**If you want exact counts:**

Use exact tokenizer (see `context_manager.py`):
```python
from transformers import AutoTokenizer
from context_manager import ExactTokenCounter

tokenizer = AutoTokenizer.from_pretrained("model-name")
counter = ExactTokenCounter(tokenizer)
```

Then configure:
```python
self.context_manager = ContextManager(
    context_window=4096,
    token_counter=counter,  # Use exact
)
```

**For now:** Approximate counter is safe and working as intended.

---

## Issue: Changes to Config Not Working

### Symptoms
- Changed appconf.json
- Config changes don't seem to apply
- Using old values

### Root Cause
Python process caching config, need restart.

### Solution

**Restart the chatbot:**
```bash
# Kill existing process
python main.py  # Start fresh
```

**Verify config loaded:**
```json
{
  "DEBUG_CONTEXT": true,
  "SHOW_LOGS": true
}
```

Should log: "Loaded config type: <class 'dict'>"

---

## Issue: "Error in Dynamic Context Allocation" Exception

### Symptoms
- Logs show error in allocation logic
- Falls back to static
- Continues working but shows warning

### Root Cause
Bug in dynamic allocation or unexpected input.

### Solution

**Check error message:**
```
[WARNING] Error in dynamic context allocation: [specific error]
```

**Common fixes:**

1. Missing config option
```json
{
  "CHUNK_SIZE_ESTIMATE": 150,
  "MAX_INITIAL_RETRIEVAL": 20
}
```

2. Disable dynamic context
```json
{
  "USE_DYNAMIC_CONTEXT": false
}
```

3. Enable debugging to see details
```json
{
  "DEBUG_CONTEXT": true,
  "LOG_LEVEL": "DEBUG"
}
```

**Report issue** if error persists:
- Include error message
- Include config
- Include model information

---

## Issue: Very Long Conversations

### Symptoms
- 30+ turns in a conversation
- Getting slower
- Model quality degrading

### Root Cause
History accumulation hitting ceiling.

### Solution

**Cap history strictly:**
```json
{
  "MAX_HISTORY_TURNS": 4
}
```

**Or start fresh:**
```bash
# Exit and restart
python main.py
```

This clears conversation history (by design).

### Verify
Each new session has fresh context.

---

## Debug: Checking Allocation

### Enable Full Debugging

```json
{
  "USE_DYNAMIC_CONTEXT": true,
  "DEBUG_CONTEXT": true,
  "SHOW_LOGS": true,
  "LOG_LEVEL": "DEBUG"
}
```

### What to Look For

**Good output:**
```
Context Window: 4096 tokens
  System Prompt: 842 tokens  
  Available: 3254 tokens
Allocation:
  Input: 187 tokens
  History: 896 tokens
  Examples: 721 tokens
  Context: 1450 tokens
  Total: 3254 / 3254
```

**Warning output:**
```
Available budget (480 tokens) too small...
Using static fallback
```

**Error output:**
```
Error in dynamic context allocation: [specific error]
Using static fallback
```

---

## Debug: Manual Testing

### Test 1: Check Config Loading
```bash
python -c "import json; print(json.load(open('configs/appconf.json')))"
```

### Test 2: Check Dynamic Context Module
```bash
python -c "from context_manager import ContextManager; print('Module OK')"
```

### Test 3: Check Conversation Manager Loads
```bash
python -c "from conversation_manager import ConversationManager; print('Module OK')"
```

### Test 4: Quick Turn Test
```python
from conversation_manager import ConversationManager

manager = ConversationManager()
print(f"Dynamic context enabled: {manager.use_dynamic_context}")
print(f"Context manager: {manager.context_manager}")
```

---

## Still Not Resolved?

1. **Check latest code** - may be fixed already
2. **Review all changes** from documentation
3. **Enable DEBUG_CONTEXT** - see actual situation
4. **Try Safe Profile** - `USE_DYNAMIC_CONTEXT: false`
5. **Restart** - clear Python cache

If still broken, collect:
- Full error message
- Your config
- Steps to reproduce
- Model/GPU info

---

## Next Steps

- **Want to understand better?** → See `02_HOW_IT_WORKS.md`
- **Configuration help?** → See `03_CONFIGURATION.md`  
- **Still stuck?** → Check `07_IMPLEMENTATION.md` for code details
