#!/usr/bin/env python3
"""Test script to demonstrate automatic GPU layer calculation."""

from pathlib import Path

from gpu_utils import calculate_optimal_layers, get_gpu_memory_info

# Test GPU detection
print("=" * 60)
print("GPU Memory Detection Test")
print("=" * 60)

gpu_info = get_gpu_memory_info()
if gpu_info:
    print(f"GPU Name: {gpu_info['gpu_name']}")
    print(f"Total VRAM: {gpu_info['total_vram_mb']:.0f} MB")
    print(f"Used VRAM: {gpu_info['used_vram_mb']:.0f} MB")
    print(f"Free VRAM: {gpu_info['free_vram_mb']:.0f} MB")
else:
    print("GPU info unavailable (fallback mode will be used)")

print("\n" + "=" * 60)
print("Automatic Layer Calculation Test")
print("=" * 60)

# Test with your actual model
model_path = Path("./models/mistral_v0.1/mistral-7b-instruct-v0.1.Q4_K_M.gguf")

if model_path.exists():
    print(f"\nModel: {model_path.name}")
    print(f"File size: {model_path.stat().st_size / (1024**2):.1f} MB")

    # Test different target VRAM usage percentages
    for target_pct in [0.6, 0.7, 0.8, 0.9]:
        result = calculate_optimal_layers(model_path, target_vram_usage=target_pct, n_ctx=32768)

        print(f"\n--- Target VRAM Usage: {target_pct * 100:.0f}% ---")
        print(f"Calculated GPU layers: {result['n_gpu_layers']}")
        print(f"Estimated VRAM usage: {result['estimated_vram_mb']:.0f} MB")
        if result["available_vram_mb"] > 0:
            actual_pct = (result["estimated_vram_mb"] / result["available_vram_mb"]) * 100
            print(f"Actual usage: {actual_pct:.1f}% of {result['available_vram_mb']:.0f} MB")
else:
    print(f"\nModel not found: {model_path}")
    print("Skipping layer calculation test")

print("\n" + "=" * 60)
print("Configuration Examples")
print("=" * 60)
print('\nIn modelconf.json, you can now use:')
print('  "LAYERS": "auto"           # Automatic detection (80% VRAM by default)')
print('  "LAYERS": -1               # Also triggers automatic detection')
print('  "LAYERS": 15               # Manual override (static value)')
print('\nYou can also customize target VRAM usage:')
print('  "TARGET_VRAM_USAGE": 0.8   # Use 80% of GPU memory (default)')
print('  "TARGET_VRAM_USAGE": 0.7   # Use 70% of GPU memory (more conservative)')
