# Automatic GPU Layer Offloading

## Overview

This chatbot now supports **automatic GPU layer calculation** instead of manually setting the `LAYERS` parameter for each model. The system detects available VRAM and calculates the optimal number of layers to offload to your GPU.

## Features

- **Automatic VRAM Detection**: Uses NVIDIA Management Library (pynvml) to query GPU memory
- **Smart Layer Calculation**: Estimates memory usage per layer based on model size and context length
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
  ...
}
```

### Configuration Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `LAYERS` | `"auto"` or `int` | Use `"auto"` for automatic detection, or specify an integer for manual control | `"auto"` |
| `TARGET_VRAM_USAGE` | `float` (0.0-1.0) | Percentage of total VRAM to target | `0.8` (80%) |

### Examples

```json
// Automatic with 80% VRAM target (recommended)
"LAYERS": "auto",
"TARGET_VRAM_USAGE": 0.8

// Automatic with 70% VRAM target (more conservative)
"LAYERS": "auto",
"TARGET_VRAM_USAGE": 0.7

// Manual override (disables automatic calculation)
"LAYERS": 20

// Also triggers automatic detection
"LAYERS": -1
```

## How It Works

1. **GPU Detection**: Queries NVIDIA GPU using pynvml to get total, used, and free VRAM
2. **Model Analysis**: Estimates memory requirements based on:
   - Model file size (determines layer count and weight size)
   - Model architecture (hidden size: 4096 for 7B, 5120 for 13B models)
   - Context window size (N_CTX) - **has major impact on VRAM**
   - KV cache per layer: `2 * N_CTX * hidden_size * 2 bytes`
3. **Layer Calculation**: Computes maximum layers that fit within target VRAM usage
4. **Safety Margins**: Includes overhead for llama.cpp runtime and buffer space

### Memory Formula Per Layer

**Each offloaded layer requires:**
- **Weights**: `model_size / total_layers` MB
- **KV Cache**: `2 * N_CTX * hidden_size * 2 / 1024²` MB

**Example (Mistral 7B Q4_K_M, 32K context):**
- Weights: ~130 MB/layer
- KV cache: ~512 MB/layer
- **Total: ~642 MB per layer**

This means context size significantly impacts how many layers fit in VRAM!

## Benefits

- **No Manual Tuning**: Automatically adjusts for different models
- **Multi-Model Support**: Switch between models without reconfiguring layers
- **Optimal Performance**: Uses maximum available VRAM while leaving headroom
- **Protection**: Prevents OOM errors by staying within target limits
- **Context-Aware**: Accounts for KV cache scaling with context window size

## Important Notes

⚠️ **Context size has a massive impact on memory usage!**

With large context windows (32K+), you may only be able to offload a handful of layers to an 8GB GPU. This is expected behavior because:
- Each layer needs KV cache: `2 * N_CTX * hidden_size * 2 bytes`
- For 32K context: ~512 MB per layer just for KV cache
- For 8K context: ~128 MB per layer for KV cache

**Realistic expectations for 8GB GPU + Mistral 7B:**
- 32K context: ~8-10 layers
- 16K context: ~14-16 layers  
- 8K context: ~20-25 layers
- 4K context: ~30-32 layers (full model)

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

**This is likely correct!** Large context windows consume significant VRAM:
- Check your `N_CTX` value in modelconf.json
- 32K context uses ~512 MB per layer just for KV cache
- Consider reducing context size if you need more layers on GPU

Example for 8GB GPU:
- `N_CTX: 32768` → ~8 layers (realistic for heavy usage)
- `N_CTX: 8192` → ~22 layers (good balance)
- `N_CTX: 4096` → ~30 layers (full model fits)

### Out of Memory Errors

If you get OOM errors even with auto-calculation:
- Lower `TARGET_VRAM_USAGE` to 0.7 or 0.6
- Reduce `N_CTX` (context window size)
- Use a smaller quantization (Q3_K_M instead of Q4_K_M)

### Too Many/Few Layers

Adjust `TARGET_VRAM_USAGE`:
- Lower value (e.g., 0.6-0.7): More conservative, leaves more free VRAM
- Higher value (e.g., 0.85-0.9): More aggressive, uses more VRAM

### Prefer Manual Control

Simply set `LAYERS` to an integer value:
```json
"LAYERS": 8
```

This completely bypasses automatic detection.
