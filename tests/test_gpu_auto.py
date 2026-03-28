#!/usr/bin/env python3
"""Test script to demonstrate automatic GPU layer calculation."""

from pathlib import Path

from core.gpu_utils import calculate_optimal_layers, get_gpu_memory_info

# Test GPU detection
print("=" * 60)  # noqa: T201
print("GPU Memory Detection Test")  # noqa: T201
print("=" * 60)  # noqa: T201

gpu_info = get_gpu_memory_info()
if gpu_info:
    print(f"GPU Name: {gpu_info['gpu_name']}")  # noqa: T201
    print(f"Total VRAM: {gpu_info['total_vram_mb']:.0f} MB")  # noqa: T201
    print(f"Used VRAM: {gpu_info['used_vram_mb']:.0f} MB")  # noqa: T201
    print(f"Free VRAM: {gpu_info['free_vram_mb']:.0f} MB")  # noqa: T201
else:
    print("GPU info unavailable (fallback mode will be used)")  # noqa: T201

print("\n" + "=" * 60)  # noqa: T201
print("Automatic Layer Calculation Test")  # noqa: T201
print("=" * 60)  # noqa: T201

# Test with your actual model
model_path = Path("./models/mistral_v0.1/mistral-7b-instruct-v0.1.Q4_K_M.gguf")

if model_path.exists():
    # Test different target VRAM usage percentages
    for target_pct in [0.6, 0.7, 0.8, 0.9]:
        result = calculate_optimal_layers(model_path, target_vram_usage=target_pct, n_ctx=32768)

        if result["available_vram_mb"] > 0:
            actual_pct = (result["estimated_vram_mb"] / result["available_vram_mb"]) * 100
else:
    pass
