# Dynamic Context Window Management

## Overview

The dynamic context management system intelligently allocates available tokens from your model's context window across multiple prompt components:

- **System Prompt**: Character description, scenario, and instructions
- **Message Examples**: Few-shot examples from the character card or RAG
- **Conversation History**: Previous exchanges in the current conversation
- **Vector Context**: Retrieved documents/context from RAG (ChromaDB)

Instead of using a fixed `RAG_K` parameter or hard-coded history limits, the system automatically scales these components based on available context.

## How It Works

### 1. Context Budget Calculation

When you send a message, the system:

1. Detects your model's context window (e.g., 4096 tokens for Llama 3.1-8B)
2. Reserves space for the model's response (configurable, default 256 tokens)
3. Calculates the base system prompt size
4. Determines remaining budget for dynamic content

```
Available Budget = Context Window - Response Reserve - System Prompt
```

### 2. Dynamic Allocation

The remaining budget is intelligently distributed:

- **User Input** (required): ~10% of dynamic budget (capped at 500 tokens)
- **Conversation History** (coherence): 30% of remaining budget
- **Message Examples** (guidance): 25% of remaining budget
- **Vector Context** (details): Remaining budget

Priority is given to coherence and structure of the prompt, ensuring quality responses.

### 3. Token Counting

Two token counting strategies are available:

- **Approximate Counter (Default)**: Fast, character-based heuristics (~1 token per 4 characters)
  - Good enough for most use cases
  - No external dependencies
  - Conservative estimates prevent overflow

- **Exact Counter**: Uses HuggingFace tokenizer for precise counts
  - More accurate but slightly slower
  - Can be integrated if you already have a tokenizer

## Configuration

Add these options to `configs/appconf.json`:

### Core Settings

```json
{
  "USE_DYNAMIC_CONTEXT": true,
  "DEBUG_CONTEXT": false,
  "RESERVED_FOR_RESPONSE": 256,
  "MIN_HISTORY_TURNS": 1,
  "MAX_HISTORY_TURNS": 8
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `USE_DYNAMIC_CONTEXT` | `true` | Enable/disable dynamic allocation. When `false`, falls back to static `RAG_K` |
| `DEBUG_CONTEXT` | `false` | Log detailed context allocation info for each turn |
| `RESERVED_FOR_RESPONSE` | `256` | Tokens reserved for model output |
| `MIN_HISTORY_TURNS` | `1` | Never drop below this many conversation turns |
| `MAX_HISTORY_TURNS` | `8` | Never include more than this many turns |

### Legacy Settings (Still Supported)

```json
{
  "RAG_K": 2,
  "N_CTX": 4096
}
```

- `RAG_K`: Used as fallback when `USE_DYNAMIC_CONTEXT=false`
- `N_CTX`: If not detected from model, used as context window size

## Usage Examples

### Example 1: Small Context Model (3.8B with 4K context)

With a Mistral 7B-Instruct Q3_K model:

```
Context Window: 4096 tokens
Response Buffer: 256 tokens
System Prompt: ~800 tokens
Available: ~3040 tokens

Allocation per turn:
- User Input: ~300 tokens
- History: ~900 tokens (recent 2-3 turns)
- Examples: ~750 tokens
- Context: ~1090 tokens
```

The system will automatically retrieve and include more RAG chunks than a 3-item limit would allow.

### Example 2: Large Context Model (70B with 8K+ context)

With a Llama 3.1-8B model running with extended context:

```
Context Window: 8192 tokens
Response Buffer: 256 tokens
System Prompt: ~800 tokens
Available: ~7136 tokens

Allocation per turn:
- User Input: ~700 tokens
- History: ~2140 tokens (most/all recent conversation)
- Examples: ~1785 tokens
- Context: ~2511 tokens
```

Much more context, examples, and history can be included.

### Example 3: Long Conversation with Limited Context

After 10+ turns in a 4K model:

```
Available for dynamic content: 3040 tokens

MIN_HISTORY_TURNS = 1 ensures at least the last exchange is kept.
MAX_HISTORY_TURNS = 8 prevents using too many old turns.
Actual allocation: ~3 turns of recent history
```

The system respects conversation flow while staying within budget.

## Debugging Context Allocation

Enable detailed logging with:

```json
{
  "DEBUG_CONTEXT": true,
  "SHOW_LOGS": true,
  "LOG_LEVEL": "DEBUG"
}
```

Output will show:

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

## Integration with Existing Code

The dynamic context system is **backward compatible**:

- If `USE_DYNAMIC_CONTEXT=false`, behaves like before using `RAG_K`
- The `ConversationManager` automatically initializes the context manager
- No changes needed to existing conversation flows

## Optimization Tips

### 1. Detect Model Context at Startup

The system automatically reads `n_ctx` from `llama-cpp-python`:

```python
# Automatic (already in code):
context_window = self._read_llama_ctx_value(client, "n_ctx")
```

If automatic detection fails, set `N_CTX` in config.

### 2. Tune Reserved Response Space

If responses get cut off: increase `RESERVED_FOR_RESPONSE` to 512+  
If you're wasting tokens: decrease to 128-200

### 3. Adjust History Balance

For better coherence with long contexts:
```json
{
  "MIN_HISTORY_TURNS": 2,
  "MAX_HISTORY_TURNS": 12
}
```

For rapid turn-taking with limited context:
```json
{
  "MIN_HISTORY_TURNS": 1,
  "MAX_HISTORY_TURNS": 3
}
```

## Advanced: Custom Token Counter

To use an exact tokenizer:

```python
from transformers import AutoTokenizer
from context_manager import ExactTokenCounter

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
counter = ExactTokenCounter(tokenizer)

# Then in ConversationManager.__init__:
self.context_manager = ContextManager(
    context_window=4096,
    token_counter=counter,
)
```

## Performance Impact

- **Startup**: +50-200ms (context manager initialization)
- **Per Turn**: +10-50ms (token counting + allocation)
- **Memory**: Minimal (<1MB additional)

Overhead is negligible compared to LLM inference time.

## Troubleshooting

### Context window not detected

**Problem**: Logs show "Could not detect context window, using default"

**Solution**:
```json
{
  "N_CTX": 4096
}
```

### History keeps getting truncated

**Problem**: Even with "MIN_HISTORY_TURNS", old turns disappear

**Solution**: Increase `RESERVED_FOR_RESPONSE` if the system is over-conservative:
```json
{
  "RESERVED_FOR_RESPONSE": 128
}
```

### Too much or too little context

**Problem**: RAG context seems too short/long

**Solution**: Check allocation with `DEBUG_CONTEXT: true` and adjust history/example ratios in `core/context_manager.py` allocation percentages (lines ~170-180).

### Token counting seems wrong

**Problem**: Estimated tokens don't match actual model output

**Solution**: The approximate counter is intentionally conservative. For exact counts, implement `ExactTokenCounter` with your model's tokenizer.

## API Reference

### ContextManager Class

```python
class ContextManager:
    def __init__(
        self,
        context_window: int,
        token_counter: TokenCounterProtocol | None = None,
        reserved_for_response: int = 256,
        min_history_turns: int = 1,
        max_history_turns: int = 8,
    ) -> None:
        """Initialize context manager."""
        
    def calculate_budget(self, system_prompt: str) -> ContextBudget:
        """Calculate available token budget."""
        
    def allocate_content(
        self,
        budget: ContextBudget,
        message_examples: str,
        vector_context: str,
        conversation_history: str,
        current_input: str,
    ) -> dict[str, str | int]:
        """Allocate tokens among components."""
```

### Integration Points in ConversationManager

```python
def _prepare_vector_context(self, message: str) -> tuple[str, str]:
    """Called before building the prompt. Returns (context, examples)."""
    if self.use_dynamic_context:
        # Dynamic allocation happens here
        
def _build_system_prompt_text(self, mes_example: str) -> str:
    """Build system prompt for budget calculation."""
```

## Future Enhancements

Potential improvements:

1. **Learning-based allocation**: Track which components most affect quality, optimize ratios
2. **Token budget persistence**: Remember successful allocations for similar topics
3. **Streaming token counter**: Count tokens as they're generated
4. **Context window specialization**: Detect model family and optimize defaults

## References

- [llamacpp-python context window detection](https://github.com/abetlen/llama-cpp-python)
- [LangChain Chroma integration](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/vectorstores)
- [Token counting strategies](https://platform.openai.com/docs/guides/tokens)
