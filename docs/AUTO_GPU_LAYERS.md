# Automatic GPU Layer Selection

Last verified: 2026-03-01

This project supports automatic `llama.cpp` GPU offload layer selection through `core/gpu_utils.py` and `ConversationManager`.

## What It Does

When `LAYERS` is set to `"auto"` in `configs/modelconf.json`, runtime:

1. Reads GPU memory info from NVML (`pynvml`).
2. Estimates model layer count and per-layer VRAM usage.
3. Applies `TARGET_VRAM_USAGE` and a runtime overhead reserve.
4. Chooses `n_gpu_layers` automatically for `LlamaCpp`.

If GPU info is unavailable, it falls back to a conservative default.

## Key Configuration

From `configs/modelconf.json`:

```json
{
  "LAYERS": "auto",
  "TARGET_VRAM_USAGE": 0.8,
  "N_CTX": 32768,
  "KV_CACHE_QUANT": "f16"
}
```

## Related Runtime Options

- `N_CTX`: larger context increases KV-cache memory usage per layer.
- `KV_CACHE_QUANT`: `f16`, `q8_0`, or `q4_0` (validated at runtime).
- `CHECK_MODEL_CONTEXT` and `AUTO_ADJUST_MODEL_CONTEXT` in `configs/appconf.json` control context-window sanity behavior.

## Quick Verification

```bash
uv run python tests/test_gpu_auto.py
```

For KV quantization analysis script:

```bash
uv run python tests/test_kv_quantization.py
```

## Code References

- `core/gpu_utils.py`
- `core/conversation_manager.py`
- `tests/test_gpu_auto.py`
- `tests/test_kv_quantization.py`
