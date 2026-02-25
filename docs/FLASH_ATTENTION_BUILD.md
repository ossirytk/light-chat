# Flash Attention Build Guide for uv

## Quick Start

For your `uv`-based project, rebuilding `llama-cpp-python` with Flash Attention is straightforward:

### Easy: Use the Automated Script

```fish
# Activate your environment
. .venv/bin/activate.fish

# Run the build script
uv run python scripts/build/flash_attention/build_flash_attention.py
```

Or with bash:
```fish
bash scripts/build/flash_attention/build_flash_attention.sh
```

Both scripts will:
- ✅ Check system dependencies (cmake, build tools)
- ✅ Install missing dependencies if needed
- ✅ Verify CUDA availability
- ✅ Build with CUDA and Flash Attention
- ✅ Test the installation

### Manual: Step by Step with uv

If you prefer to do it manually:

```fish
# 1. Activate your environment
. .venv/bin/activate.fish

# 2. Install build tools
uv pip install cmake scikit-build-core

# 3. Set build options
set -x CMAKE_ARGS "-DGGML_CUDA=ON -DGGML_FLASH_ATTN=ON"

# 4. Rebuild llama-cpp-python
uv pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python

# 5. Test
uv run python -c "from llama_cpp import Llama; print('✓ Build successful')"

# 6. Run your app
uv run python main.py
```

## Build Options Explained

The `CMAKE_ARGS` environment variable controls the build:

- **`-DGGML_CUDA=ON`**: Enables NVIDIA GPU acceleration
- **`-DGGML_FLASH_ATTN=ON`**: Enables Flash Attention optimization

### Option Combinations

```fish
# Full optimization (recommended for 8GB GPU)
set -x CMAKE_ARGS "-DGGML_CUDA=ON -DGGML_FLASH_ATTN=ON"

# Just CUDA (if Flash Attention fails)
set -x CMAKE_ARGS "-DGGML_CUDA=ON"

# CPU only (fallback)
set -e CMAKE_ARGS
# or: set -x CMAKE_ARGS ""
```

## Why This Matters for Your 8GB GPU

Your current setup with auto-layer detection gets you:
- **9 layers on GPU** with 32K context (full precision KV)
- **15 layers on GPU** with 32K context (8-bit KV quantization)
- **17 layers on GPU** with Flash Attention + 8-bit KV

That's **+89% improvement** in GPU computation!

## Performance Impact

Approximate speedup with all optimizations:

| Feature | Impact |
|---------|--------|
| Base (no GPU layers) | 1.0x |
| + Layer offloading (9 layers) | 1.3x |
| + KV quantization (15 layers) | 1.5x |
| + Flash Attention (17 layers) | 2.0-2.5x |

## System Requirements

Before building, ensure you have:

```fish
# Check NVIDIA driver
nvidia-smi

# Check CUDA toolkit
nvcc --version

# Check cmake availability
cmake --version
```

If any are missing:
```fish
# Install build tools
sudo apt-get update
sudo apt-get install -y cmake build-essential

# Install CUDA (if not present)
# Follow: https://docs.nvidia.com/cuda/wsl-user-guide/
```

## Troubleshooting

### Build fails with "CUDA not found"

```fish
# Check if CUDA is Path
which nvcc

# If not found, add CUDA to PATH
set -x PATH /usr/local/cuda/bin $PATH
set -x LD_LIBRARY_PATH /usr/local/cuda/lib64 $LD_LIBRARY_PATH

# Try again
uv pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```

### Build takes too long or fails with memory errors

```fish
# Clear pip cache
uv pip cache purge

# Try CUDA only (no Flash Attention)
set -x CMAKE_ARGS "-DGGML_CUDA=ON"
uv pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```

### "Build tools not found"

The automated scripts will offer to install cmake and build-essential. If manual:

```fish
# Install build essentials
sudo apt-get install -y cmake build-essential

# On Ubuntu 22.04+, may need:
sudo apt-get install -y gcc g++ git
```

## Verifying the Build

After successful build, verify with:

```fish
# Check import
uv run python -c "from llama_cpp import Llama; print('✓')"

# Run your app
uv run python main.py

# Check auto-layer detection works
uv run python tests/test_gpu_auto.py
```

## After Building

Your app will automatically use:
- CUDA for GPU computation
- Flash Attention for optimized attention mechanism
- Automatic layer calculation for your GPU
- KV cache quantization (if configured)

No additional configuration needed! Just run:

```fish
uv run python main.py
```

## For Comparison: Traditional pip Approach

If you were using pip instead of uv, you'd do:

```fish
# Traditional pip (NOT recommended, use uv instead)
set -x CMAKE_ARGS "-DGGML_CUDA=ON -DGGML_FLASH_ATTN=ON"
pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```

With `uv`, the extra tools come from the uv ecosystem, making the build more reproducible!
