# Automatic GPU Layer Offloading

## Overview

This chatbot now supports **automatic GPU layer calculation** and **KV cache quantization** to optimize VRAM usage. The system detects available VRAM and intelligently balances layer offloading with memory constraints.

## Features

- **Automatic VRAM Detection**: Uses NVIDIA Management Library (pynvml) to query GPU memory
- **Smart Layer Calculation**: Estimates memory usage per layer based on model size and context length
- **KV Cache Quantization**: Reduces KV cache memory by 50-75% with minimal quality loss
- **Configurable Target Usage**: Set your preferred VRAM usage percentage (default: 80%)
- **Fallback Support**: Works even without GPU detection (uses conservative defaults)
- **Per-Model Optimization**: Automatically adjusts for different model sizes and quantizations

## Configuration

### modelconf.json

```json
{
  "MODEL": "mistral_v0.1/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
  "MODEL_TYPE": "mistral",
  "LAYERS": "auto",
  "TARGET_VRAM_USAGE": 0.8,
  "KV_CACHE_QUANT": "f16",
  ...
}
```

### Configuration Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `LAYERS` | `"auto"` or `int` | Use `"auto"` for automatic detection, or specify an integer for manual control | `"auto"` |
| `TARGET_VRAM_USAGE` | `float` (0.0-1.0) | Percentage of total VRAM to target | `0.8` (80%) |
| `KV_CACHE_QUANT` | `"f16"`, `"q8_0"`, `"q4_0"` | KV cache quantization (see below) | `"f16"` (full precision) |

### Examples

```json
// Automatic with 80% VRAM target + full precision KV (default)
"LAYERS": "auto",
"TARGET_VRAM_USAGE": 0.8,
"KV_CACHE_QUANT": "f16"

// Automatic with 8-bit KV quantization (better performance)
"LAYERS": "auto",
"TARGET_VRAM_USAGE": 0.8,
"KV_CACHE_QUANT": "q8_0"

// Maximum compression: 8K context + 4-bit KV (full model on 8GB GPU!)
"N_CTX": 8192,
"KV_CACHE_QUANT": "q4_0",
"LAYERS": "auto"

// Manual override (disables automatic calculation)
"LAYERS": 15
```

## How It Works

1. **GPU Detection**: Queries NVIDIA GPU using pynvml to get total, used, and free VRAM
2. **Model Analysis**: Estimates memory requirements based on:
   - Model file size (determines layer count and weight size)
   - Model architecture (hidden size: 4096 for 7B, 5120 for 13B models)
   - Context window size (N_CTX) - **has major impact on VRAM**
   - KV cache per layer: `2 * N_CTX * hidden_size * bytes_per_element`
   - KV cache quantization (if enabled)
3. **Layer Calculation**: Computes maximum layers that fit within target VRAM usage
4. **Safety Margins**: Includes overhead for llama.cpp runtime and buffer space

### Memory Formula Per Layer

**Without KV quantization (fp16):**
- **Weights**: `model_size / total_layers` MB
- **KV Cache**: `2 * N_CTX * hidden_size * 2 / 1024²` MB

**With KV quantization:**
- **Weights**: Same as above
- **KV Cache (q8_0)**: 50% of fp16
- **KV Cache (q4_0)**: 25% of fp16

### Example: Mistral 7B Q4_K_M

**Without KV quantization (fp16):**
- Weights: ~130 MB/layer
- KV cache (32K context): ~512 MB/layer
- **Total: ~642 MB per layer → 8-9 layers fit**

**With 8-bit KV quantization (q8_0):**
- Weights: ~130 MB/layer
- KV cache (32K context): ~256 MB/layer
- **Total: ~386 MB per layer → 15-16 layers fit (+67%!)**

**With 4-bit KV quantization (q4_0):**
- Weights: ~130 MB/layer
- KV cache (32K context): ~128 MB/layer
- **Total: ~258 MB per layer → 23-24 layers fit (+156%!)**

## Benefits

- **No Manual Tuning**: Automatically adjusts for different models
- **Multi-Model Support**: Switch between models without reconfiguring layers
- **Optimal Performance**: Uses maximum available VRAM while leaving headroom
- **Protection**: Prevents OOM errors by staying within target limits
- **Context-Aware**: Accounts for KV cache scaling with context window size
- **KV Quantization Support**: Reduce KV cache memory by 50-75% with minimal quality loss

## Important Notes

⚠️ **Context size has a massive impact on memory usage!**

With large context windows (32K+), KV cache quantization is highly recommended:
- Each layer needs KV cache: `2 * N_CTX * hidden_size * bytes`
- For 32K context fp16: ~512 MB per layer just for KV cache
- For 32K context q8_0: ~256 MB per layer (50% savings!)
- For 32K context q4_0: ~128 MB per layer (75% savings!)

**Realistic expectations for 8GB GPU + Mistral 7B:**
- 32K context + fp16: ~8-10 layers
- 32K context + q8_0: ~15-16 layers (2x improvement!)
- 32K context + q4_0: ~23-24 layers (2.5x improvement!)
- 8K context + f16: ~20-25 layers
- 4K context + f16: ~30-32 layers (full model)

## Testing

Run the test script to see automatic detection in action:

```bash
uv run python test_gpu_auto.py
```

Example output (Mistral 7B Q4_K_M with 32K context on 8GB GPU):
```
GPU Memory Detection Test
============================================================
GPU Name: Quadro P4000
Total VRAM: 8192 MB
Used VRAM: 857 MB
Free VRAM: 7335 MB

Automatic Layer Calculation Test
============================================================
Model: mistral-7b-instruct-v0.1.Q4_K_M.gguf
File size: 4166.1 MB
VRAM per layer: 130.2 MB (weights) + 512.0 MB (KV cache @ 32768) = 642.2 MB

--- Target VRAM Usage: 80% ---
Calculated GPU layers: 8
Estimated VRAM usage: 5638 MB
Actual usage: 68.8% of 8192 MB
```

### Impact of Context Size

The same model with different context sizes on 8GB GPU:

| N_CTX | KV Cache/Layer | Layers @ 80% | Total VRAM |
|-------|----------------|--------------|------------|
| 2048 | 32 MB | ~32 (full) | ~5.2 GB |
| 4096 | 64 MB | ~30 (full) | ~5.8 GB |
| 8192 | 128 MB | ~22 | ~5.7 GB |
| 16384 | 256 MB | ~14 | ~5.4 GB |
| **32768** | **512 MB** | **8** | **5.6 GB** |

**Key Takeaway:** Large context windows drastically reduce the number of layers that fit in VRAM!

## Dependencies

The automatic detection requires `nvidia-ml-py`:
```bash
uv add nvidia-ml-py
```

If the package is not installed, the system will fall back to conservative defaults.

## Migration from Manual Configuration

Your old configuration:
```json
"LAYERS": 15
```

Simply change to:
```json
"LAYERS": "auto",
"TARGET_VRAM_USAGE": 0.8
```

The system will automatically calculate the optimal number of layers for your GPU and model.

## Troubleshooting

### GPU Not Detected

If you see "GPU info unavailable", check:
- NVIDIA drivers are installed: `nvidia-smi`
- `nvidia-ml-py` is installed: `uv add nvidia-ml-py`
- Running on WSL/Ubuntu with CUDA enabled

### Only Getting a Few Layers?

**This is likely correct, but try KV quantization!**
- Check your `N_CTX` value in modelconf.json
- Enable KV quantization (`q8_0` or `q4_0`) to fit more layers
- 32K context + fp16 = only 8-9 layers feasible with 8GB GPU
- 32K context + q8_0 = 15-16 layers (2x improvement!)
- 32K context + q4_0 = 23-24 layers (2.5x improvement!)

### Recommended Settings for 8GB GPU

**Option A: Keep 32K context (better conversation history)**
```json
"N_CTX": 32768,
"KV_CACHE_QUANT": "q8_0"
```
Result: ~15 layers, minimal quality loss

**Option B: Reduce context (simpler, full model on GPU)**
```json
"N_CTX": 8192,
"KV_CACHE_QUANT": "f16"
```
Result: ~23 layers, full model fits on GPU

**Option C: Maximum performance**
```json
"N_CTX": 8192,
"KV_CACHE_QUANT": "q4_0"
```
Result: ~32 layers (entire model on GPU)

### Out of Memory Errors

If you get OOM errors even with auto-calculation:
1. Enable KV cache quantization: `"KV_CACHE_QUANT": "q8_0"`
2. Lower `TARGET_VRAM_USAGE` to 0.7 or 0.6
3. Reduce `N_CTX` (context window size)
4. Use a smaller quantization (Q3_K_M instead of Q4_K_M)

### Prefer Manual Control

Simply set `LAYERS` to an integer value:
```json
"LAYERS": 15
```

This completely bypasses automatic detection.
