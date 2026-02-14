"""GPU memory utilities for automatic layer offloading calculation."""

from pathlib import Path
from typing import TypedDict

from loguru import logger

# Model size thresholds for layer estimation (MB)
MODEL_SIZE_7B_THRESHOLD = 5000
MODEL_SIZE_13B_THRESHOLD = 9000


class GPUMemoryInfo(TypedDict):
    """Information about GPU memory."""

    total_vram_mb: float
    free_vram_mb: float
    used_vram_mb: float
    gpu_name: str


class LayerCalculationResult(TypedDict):
    """Result of automatic layer calculation."""

    n_gpu_layers: int
    estimated_vram_mb: float
    available_vram_mb: float
    target_usage_pct: float
    calculation_method: str


def get_gpu_memory_info() -> GPUMemoryInfo | None:
    """
    Get current GPU memory information using pynvml.

    Returns:
        GPUMemoryInfo dict with VRAM details, or None if CUDA is unavailable
    """
    try:
        import pynvml  # noqa: PLC0415

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)

        gpu_info: GPUMemoryInfo = {
            "total_vram_mb": info.total / (1024**2),
            "free_vram_mb": info.free / (1024**2),
            "used_vram_mb": info.used / (1024**2),
            "gpu_name": name if isinstance(name, str) else name.decode("utf-8"),
        }

        pynvml.nvmlShutdown()
    except ImportError:
        logger.warning("pynvml not available. Install with: uv add nvidia-ml-py")
        return None
    except Exception as e:
        logger.warning("Failed to get GPU info: {}", e)
        return None
    else:
        return gpu_info


def estimate_model_params(model_path: Path) -> tuple[int, int]:
    """
    Estimate model parameters from file size and filename.

    Args:
        model_path: Path to the GGUF model file

    Returns:
        Tuple of (estimated_layers, hidden_size)
    """
    model_size_mb = model_path.stat().st_size / (1024**2)
    filename = model_path.name.lower()

    # Estimate hidden size from filename or size
    # 7B models typically have hidden_size of 4096
    # 13B models typically have hidden_size of 5120
    if "13b" in filename or (MODEL_SIZE_7B_THRESHOLD <= model_size_mb < MODEL_SIZE_13B_THRESHOLD):
        estimated_layers = 40
        hidden_size = 5120
    elif "7b" in filename or model_size_mb < MODEL_SIZE_7B_THRESHOLD:
        estimated_layers = 32
        hidden_size = 4096
    else:  # Larger models (70B+)
        estimated_layers = 80
        hidden_size = 8192

    return estimated_layers, hidden_size


def estimate_model_vram_per_layer(model_path: Path, n_ctx: int = 2048) -> tuple[float, int]:
    """
    Estimate VRAM usage per layer based on model file size and context length.

    Args:
        model_path: Path to the GGUF model file
        n_ctx: Context window size (affects KV cache size)

    Returns:
        Tuple of (VRAM in MB per layer, estimated total layers)
    """
    if not model_path.exists():
        logger.error("Model file not found: {}", model_path)
        return 50.0, 32  # Default fallback

    model_size_mb = model_path.stat().st_size / (1024**2)
    estimated_layers, hidden_size = estimate_model_params(model_path)

    logger.debug(
        "Model file size: {:.1f} MB | Estimated: {} layers, hidden_size={}",
        model_size_mb,
        estimated_layers,
        hidden_size,
    )

    # Base VRAM per layer (weights only)
    base_vram_per_layer = model_size_mb / estimated_layers

    # KV cache per layer for FULL context window
    # KV cache = 2 (k and v) * n_ctx * hidden_size * bytes_per_element
    # Using fp16 (2 bytes) for KV cache
    kv_cache_per_layer_mb = (2 * n_ctx * hidden_size * 2) / (1024**2)

    vram_per_layer = base_vram_per_layer + kv_cache_per_layer_mb

    logger.debug(
        "VRAM per layer: {:.1f} MB (weights) + {:.1f} MB (KV cache @ {}) = {:.1f} MB",
        base_vram_per_layer,
        kv_cache_per_layer_mb,
        n_ctx,
        vram_per_layer,
    )

    return vram_per_layer, estimated_layers


def calculate_optimal_layers(
    model_path: Path,
    target_vram_usage: float = 0.8,
    n_ctx: int = 2048,
    overhead_mb: float = 500.0,
) -> LayerCalculationResult:
    """
    Calculate optimal number of GPU layers based on available VRAM.

    Args:
        model_path: Path to the GGUF model file
        target_vram_usage: Target percentage of VRAM to use (0.0-1.0)
        n_ctx: Context window size
        overhead_mb: Base overhead for llama.cpp runtime (MB)

    Returns:
        LayerCalculationResult with calculated layers and diagnostics
    """
    gpu_info = get_gpu_memory_info()

    if gpu_info is None:
        logger.warning("GPU info unavailable, using conservative default of 20 layers")
        return LayerCalculationResult(
            n_gpu_layers=20,
            estimated_vram_mb=0.0,
            available_vram_mb=0.0,
            target_usage_pct=target_vram_usage * 100,
            calculation_method="fallback_no_gpu_info",
        )

    logger.info(
        "GPU: {} | Total: {:.0f} MB | Used: {:.0f} MB | Free: {:.0f} MB",
        gpu_info["gpu_name"],
        gpu_info["total_vram_mb"],
        gpu_info["used_vram_mb"],
        gpu_info["free_vram_mb"],
    )

    # Calculate available VRAM based on target usage
    available_vram = gpu_info["total_vram_mb"] * target_vram_usage - gpu_info["used_vram_mb"]

    # Subtract overhead
    available_for_layers = max(available_vram - overhead_mb, 0)

    logger.debug("Available VRAM for layers: {:.0f} MB", available_for_layers)

    # Estimate VRAM per layer and get model layer count
    vram_per_layer, max_layers = estimate_model_vram_per_layer(model_path, n_ctx)

    # Calculate number of layers that fit
    n_gpu_layers = int(available_for_layers / vram_per_layer)

    # Clamp to model's actual layer count (can't offload more layers than the model has)
    n_gpu_layers = max(0, min(n_gpu_layers, max_layers))

    if n_gpu_layers >= max_layers:
        logger.info("Full model offload: all {} layers fit in VRAM", max_layers)
    elif n_gpu_layers == 0:
        logger.warning("No GPU layers: model too large or VRAM insufficient")

    estimated_vram = n_gpu_layers * vram_per_layer + overhead_mb

    logger.info(
        "Calculated {} GPU layers (estimated {:.0f} MB VRAM usage, {:.1f}% of total)",
        n_gpu_layers,
        estimated_vram,
        (estimated_vram / gpu_info["total_vram_mb"]) * 100,
    )

    return LayerCalculationResult(
        n_gpu_layers=n_gpu_layers,
        estimated_vram_mb=estimated_vram,
        available_vram_mb=gpu_info["total_vram_mb"],
        target_usage_pct=target_vram_usage * 100,
        calculation_method="automatic_calculation",
    )


def get_n_gpu_layers(
    model_path: Path | str,
    configured_layers: int | str,
    n_ctx: int = 2048,
    target_vram_usage: float = 0.8,
) -> int:
    """
    Get the number of GPU layers to use, either from config or automatic calculation.

    Args:
        model_path: Path to the GGUF model file
        configured_layers: Value from config (int or "auto")
        n_ctx: Context window size
        target_vram_usage: Target VRAM usage as percentage (0.0-1.0)

    Returns:
        Number of GPU layers to offload
    """
    # Handle string "auto" or negative values as trigger for automatic calculation
    if configured_layers == "auto" or (isinstance(configured_layers, int) and configured_layers < 0):
        path = Path(model_path) if isinstance(model_path, str) else model_path
        result = calculate_optimal_layers(path, target_vram_usage, n_ctx)
        return result["n_gpu_layers"]

    # Otherwise use configured value
    return int(configured_layers)


def calculate_kv_cache_memory_saved(
    n_ctx: int,
    hidden_size: int,
    quantization_type: str = "f16",
) -> float:
    """
    Calculate VRAM saved by KV cache quantization compared to fp16.

    Args:
        n_ctx: Context window size
        hidden_size: Model hidden size
        quantization_type: KV cache quantization ("f16", "q8_0", "q4_0")

    Returns:
        Memory saved in MB per layer
    """
    # Full precision KV cache (fp16)
    full_precision_bytes_per_element = 2

    # Different quantizations
    quantization_bytes = {
        "f16": 2,  # Full precision (baseline)
        "q8_0": 1,  # 8-bit quantization
        "q4_0": 0.5,  # 4-bit quantization
    }

    bytes_per_element = quantization_bytes.get(quantization_type, 2)

    # KV cache size = 2 (k and v) * n_ctx * hidden_size * bytes_per_element
    baseline_mb = (2 * n_ctx * hidden_size * full_precision_bytes_per_element) / (1024**2)
    quantized_mb = (2 * n_ctx * hidden_size * bytes_per_element) / (1024**2)

    return baseline_mb - quantized_mb


def estimate_layers_with_kv_quantization(
    model_path: Path,
    available_vram_mb: float,
    kv_quantization: str = "f16",
    n_ctx: int = 2048,
    overhead_mb: float = 500.0,
) -> tuple[int, float]:
    """
    Calculate how many layers fit with KV cache quantization.

    Args:
        model_path: Path to model
        available_vram_mb: Available VRAM in MB
        kv_quantization: KV cache quantization type ("f16", "q8_0", "q4_0")
        n_ctx: Context window size
        overhead_mb: Runtime overhead

    Returns:
        Tuple of (number of layers, total VRAM used in MB)
    """
    _, hidden_size = estimate_model_params(model_path)
    vram_per_layer, max_layers = estimate_model_vram_per_layer(model_path, n_ctx)

    if kv_quantization != "f16":
        # Recalculate KV cache with quantization
        quantization_bytes = {
            "q8_0": 1,
            "q4_0": 0.5,
        }
        bytes_per_element = quantization_bytes.get(kv_quantization, 2)
        kv_cache_quantized = (2 * n_ctx * hidden_size * bytes_per_element) / (1024**2)

        # Base KV cache calculation (from original)
        base_vram_per_layer = model_path.stat().st_size / (1024**2) / max_layers
        vram_per_layer = base_vram_per_layer + kv_cache_quantized

        logger.debug(
            "KV quantization ({}): {:.1f} MB per layer (vs {:.1f} MB without)",
            kv_quantization,
            vram_per_layer,
            base_vram_per_layer + (2 * n_ctx * hidden_size * 2) / (1024**2),
        )

    available_for_layers = max(available_vram_mb - overhead_mb, 0)
    n_gpu_layers = min(int(available_for_layers / vram_per_layer), max_layers)
    total_vram = n_gpu_layers * vram_per_layer + overhead_mb

    return n_gpu_layers, total_vram
