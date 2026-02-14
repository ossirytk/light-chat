# Flash Attention Build Guide for uv

## Quick Start

For your `uv`-based project, rebuilding `llama-cpp-python` with Flash Attention is straightforward:

### Easy: Use the Automated Script

```bash
# Activate your environment
source .venv/bin/activate

# Run the build script
uv run python build_flash_attention.py
```

Or with bash:
```bash
bash build_flash_attention.sh
```

Both scripts will:
- ✅ Check system dependencies (cmake, build tools)
- ✅ Install missing dependencies if needed
- ✅ Verify CUDA availability
- ✅ Build with CUDA and Flash Attention
- ✅ Test the installation

### Manual: Step by Step with uv

If you prefer to do it manually:

```bash
# 1. Activate your environment
source .venv/bin/activate

# 2. Install build tools
uv pip install cmake scikit-build-core

# 3. Set build options
export CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_FLASH_ATTN=ON"

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

```bash
# Full optimization (recommended for 8GB GPU)
export CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_FLASH_ATTN=ON"

# Just CUDA (if Flash Attention fails)
export CMAKE_ARGS="-DGGML_CUDA=ON"

# CPU only (fallback)
unset CMAKE_ARGS
# or: export CMAKE_ARGS=""
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

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA toolkit
nvcc --version

# Check cmake availability
cmake --version
```

If any are missing:
```bash
# Install build tools
sudo apt-get update
sudo apt-get install -y cmake build-essential

# Install CUDA (if not present)
# Follow: https://docs.nvidia.com/cuda/wsl-user-guide/
```

## Troubleshooting

### Build fails with "CUDA not found"

```bash
# Check if CUDA is Path
which nvcc

# If not found, add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Try again
uv pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```

### Build takes too long or fails with memory errors

```bash
# Clear pip cache
uv pip cache purge

# Try CUDA only (no Flash Attention)
export CMAKE_ARGS="-DGGML_CUDA=ON"
uv pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```

### "Build tools not found"

The automated scripts will offer to install cmake and build-essential. If manual:

```bash
# Install build essentials
sudo apt-get install -y cmake build-essential

# On Ubuntu 22.04+, may need:
sudo apt-get install -y gcc g++ git
```

## Verifying the Build

After successful build, verify with:

```bash
# Check import
uv run python -c "from llama_cpp import Llama; print('✓')"

# Run your app
uv run python main.py

# Check auto-layer detection works
uv run python test_gpu_auto.py
```

## After Building

Your app will automatically use:
- CUDA for GPU computation
- Flash Attention for optimized attention mechanism
- Automatic layer calculation for your GPU
- KV cache quantization (if configured)

No additional configuration needed! Just run:

```bash
uv run python main.py
```

## For Comparison: Traditional pip Approach

If you were using pip instead of uv, you'd do:

```bash
# Traditional pip (NOT recommended, use uv instead)
CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_FLASH_ATTN=ON" pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```

With `uv`, the extra tools come from the uv ecosystem, making the build more reproducible!
