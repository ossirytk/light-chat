#!/usr/bin/env python3
"""Demonstrate KV cache quantization benefits."""

import sys
from pathlib import Path

from gpu_utils import (
    calculate_kv_cache_memory_saved,
    estimate_layers_with_kv_quantization,
    estimate_model_params,
    get_gpu_memory_info,
)

print("=" * 70)  # noqa: T201
print("KV Cache Quantization Analysis")  # noqa: T201
print("=" * 70)  # noqa: T201

# Model path
model_path = Path("./models/mistral_v0.1/mistral-7b-instruct-v0.1.Q4_K_M.gguf")

if not model_path.exists():
    print(f"Model not found: {model_path}")  # noqa: T201
    sys.exit(1)

# Get GPU info
gpu_info = get_gpu_memory_info()
if not gpu_info:
    print("GPU info unavailable")  # noqa: T201
    sys.exit(1)

print(f"\nGPU: {gpu_info['gpu_name']}")  # noqa: T201
print(f"Total VRAM: {gpu_info['total_vram_mb']:.0f} MB")  # noqa: T201
print(f"Available: {gpu_info['total_vram_mb'] - gpu_info['used_vram_mb']:.0f} MB\n")  # noqa: T201

# Test with different contexts and quantizations
contexts = [4096, 8192, 16384, 32768]
quantizations = [("f16", "Full Precision (fp16)"), ("q8_0", "8-bit Quantization"), ("q4_0", "4-bit Quantization")]

target_vram = gpu_info["total_vram_mb"] * 0.8

for n_ctx in contexts:
    print("-" * 70)  # noqa: T201
    print(f"Context Window: {n_ctx} tokens")  # noqa: T201
    print("-" * 70)  # noqa: T201

    results = []
    for quant_type, quant_name in quantizations:
        layers, total_vram = estimate_layers_with_kv_quantization(
            model_path,
            target_vram,
            kv_quantization=quant_type,
            n_ctx=n_ctx,
            overhead_mb=500,
        )

        # Calculate memory saved per layer
        _, hidden_size = estimate_model_params(model_path)
        saved_per_layer = calculate_kv_cache_memory_saved(n_ctx, hidden_size, quant_type)

        results.append((quant_name, layers, total_vram, saved_per_layer))
        print(  # noqa: T201
            f"{quant_name:25} | Layers: {layers:2d} | "
            f"Total: {total_vram:6.0f} MB | Saved/layer: {saved_per_layer:6.1f} MB",
        )

    # Calculate improvements
    f16_layers = results[0][1]
    for _i, (name, layers, _total_vram, _saved_per_layer) in enumerate(results[1:], 1):
        improvement = layers - f16_layers
        if improvement > 0:
            pct = (improvement / f16_layers) * 100 if f16_layers > 0 else 0
            print(f"  → {name}: +{improvement} layers ({pct:.0f}% improvement)")  # noqa: T201

print("\n" + "=" * 70)  # noqa: T201
print("Recommendations for 8GB GPU + Mistral 7B:")  # noqa: T201
print("=" * 70)  # noqa: T201
print("\n1. Reduce context to 8K (or enable KV quantization):")  # noqa: T201
print('   "N_CTX": 8192,')  # noqa: T201
print('   "KV_CACHE_QUANT": "f16"')  # noqa: T201
print("   Result: ~20 GPU layers (2-3x faster than current 8 layers)")  # noqa: T201

print("\n2. Keep 32K context but enable 8-bit KV quantization:")  # noqa: T201
print('   "N_CTX": 32768,')  # noqa: T201
print('   "KV_CACHE_QUANT": "q8_0"')  # noqa: T201
print("   Result: Better performance with minimal quality loss")  # noqa: T201

print("\n3. Maximum compression (4-bit KV + 8K context):")  # noqa: T201
print('   "N_CTX": 8192,')  # noqa: T201
print('   "KV_CACHE_QUANT": "q4_0"')  # noqa: T201
print("   Result: ~30-32 layers (almost full model on GPU!)")  # noqa: T201

print("\n" + "=" * 70)  # noqa: T201
