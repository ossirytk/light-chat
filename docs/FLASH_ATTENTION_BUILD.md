# Flash Attention Build Guide

Last verified: 2026-03-01

This repo includes build helpers for rebuilding `llama-cpp-python` with CUDA + Flash Attention.

## Build Helpers

- Python entry: `scripts/build/flash_attention/build_flash_attention.py`
- Shell entry: `scripts/build/flash_attention/build_flash_attention.sh`
- Compatibility wrappers:
  - `scripts/build_flash_attention.py`
  - `scripts/build_flash_attention.sh`

## Prerequisites

- Linux/WSL with NVIDIA driver + CUDA toolkit available.
- Active project environment (`uv sync` already run).
- Build tools (`cmake`, compiler toolchain).

## Recommended Command

```bash
uv run python scripts/build_flash_attention.py
```

The script rebuilds with:

```bash
CMAKE_ARGS='-DGGML_CUDA=ON -DGGML_FLASH_ATTN=ON'
```

## Verify Import

```bash
uv run python -c "from llama_cpp import Llama; print('ok')"
```

## Notes

- Performance impact varies by GPU, model, quantization, and context size.
- This document intentionally avoids fixed speedup claims.
- If Flash Attention fails, rebuild with CUDA only by adjusting `CMAKE_ARGS`.

## Related Files

- `scripts/build/flash_attention/build_flash_attention.py`
- `scripts/build/flash_attention/build_flash_attention.sh`
- `core/gpu_utils.py`
