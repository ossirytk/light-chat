# Automatic GPU Layer Selection

Last verified: 2026-03-12

This project supports automatic `llama.cpp` GPU offload layer selection through `core/gpu_utils.py` and the model-setup path used by `ConversationManager`.

## What It Does

When `model.layers` is set to `"auto"` in `configs/config.v2.json`, runtime:

1. Reads GPU memory info from NVML (`pynvml`).
2. Estimates model layer count and per-layer VRAM usage.
3. Applies `TARGET_VRAM_USAGE` and a runtime overhead reserve.
4. Chooses `n_gpu_layers` automatically for `LlamaCpp`.

If GPU info is unavailable, it falls back to a conservative default.

## Key Configuration

From `configs/config.v2.json`:

```json
{
  "model": {
    "layers": "auto",
    "target_vram_usage": 0.8,
    "n_ctx": 32768,
    "kv_cache_quant": "f16"
  }
}
```

## Related Runtime Options

- `N_CTX`: larger context increases KV-cache memory usage per layer.
- `KV_CACHE_QUANT`: `f16`, `q8_0`, or `q4_0` (validated at runtime).
- `model.context.check` and `model.context.auto_adjust` in `configs/config.v2.json` control context-window sanity behavior.

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
- `core/conversation_model_setup_mixin.py`
- `tests/test_gpu_auto.py`
- `tests/test_kv_quantization.py`
