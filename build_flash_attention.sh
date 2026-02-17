#!/bin/bash
# Build llama-cpp-python with CUDA and Flash Attention support
# For use with uv virtual environment
#
# Usage: bash build_flash_attention.sh

set -e

echo "=========================================="
echo "Building llama-cpp-python with Flash Attention"
echo "=========================================="

# Check if running in uv venv
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Not in a Python virtual environment!"
    echo "Please run: source .venv/bin/activate"
    exit 1
fi

echo "✓ Virtual environment detected: $VIRTUAL_ENV"
echo ""

# Check for required build tools
echo "Checking for build dependencies..."
if ! command -v cmake &> /dev/null; then
    echo "❌ cmake not found. Installing with apt..."
    sudo apt-get update
    sudo apt-get install -y cmake build-essential
fi

if ! command -v python &> /dev/null; then
    echo "❌ python not found in PATH"
    exit 1
fi

echo "✓ Build dependencies available"
echo ""

# Install build packages via uv
echo "Installing Python build tools with uv..."
uv pip install cmake scikit-build-core

echo ""
echo "=========================================="
echo "Building llama-cpp-python..."
echo "=========================================="
echo ""
echo "CMAKE_ARGS: -DGGML_CUDA=ON -DGGML_FLASH_ATTN=ON"
echo ""

# Set environment variables and rebuild
export CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_FLASH_ATTN=ON"

# Build with uv
uv pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python

echo ""
echo "=========================================="
echo "✓ Build Complete!"
echo "=========================================="
echo ""
echo "Testing installation..."
python -c "from llama_cpp import Llama; print('✓ llama-cpp-python imported successfully')"
echo ""
echo "To verify Flash Attention support, run:"
echo "  uv run python -c \"from llama_cpp import __version__; print(__version__)\""
