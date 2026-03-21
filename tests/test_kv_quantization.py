"""Tests for KV cache quantization estimation helpers."""

import unittest
from pathlib import Path

from core.gpu_utils import (
    calculate_kv_cache_memory_saved,
    estimate_layers_with_kv_quantization,
    estimate_model_params,
    get_gpu_memory_info,
)

MODEL_PATH = Path("./models/mistral_v0.1/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
CONTEXTS = [4096, 8192, 16384, 32768]
QUANTIZATIONS = ["f16", "q8_0", "q4_0"]


def _get_quantization_prerequisites() -> dict[str, float | str] | None:
    if not MODEL_PATH.exists():
        return None

    gpu_info = get_gpu_memory_info()
    if not gpu_info:
        return None

    return gpu_info


class TestKvQuantization(unittest.TestCase):
    """Validate monotonic improvements from more aggressive KV quantization."""

    def test_kv_quantization_allows_equal_or_more_layers_than_f16(self) -> None:
        gpu_info = _get_quantization_prerequisites()
        if gpu_info is None:
            self.skipTest("Model or GPU prerequisites unavailable")

        target_vram = float(gpu_info["total_vram_mb"]) * 0.8
        _, hidden_size = estimate_model_params(MODEL_PATH)

        for n_ctx in CONTEXTS:
            layer_counts: dict[str, int] = {}
            saved_per_layer: dict[str, float] = {}

            for quantization in QUANTIZATIONS:
                layers, total_vram = estimate_layers_with_kv_quantization(
                    MODEL_PATH,
                    target_vram,
                    kv_quantization=quantization,
                    n_ctx=n_ctx,
                    overhead_mb=500,
                )
                layer_counts[quantization] = layers
                saved_per_layer[quantization] = calculate_kv_cache_memory_saved(n_ctx, hidden_size, quantization)

                self.assertGreaterEqual(layers, 0)
                self.assertGreater(total_vram, 0)
                self.assertGreaterEqual(saved_per_layer[quantization], 0)

            self.assertGreaterEqual(layer_counts["q8_0"], layer_counts["f16"])
            self.assertGreaterEqual(layer_counts["q4_0"], layer_counts["q8_0"])
