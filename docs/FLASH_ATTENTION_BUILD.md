# Flash Attention Build Guide

Last verified: 2026-03-07

This repo includes build helpers for rebuilding `llama-cpp-python` with CUDA + Flash Attention.

## Build Helpers

- Python entry: `scripts/build/flash_attention/build_flash_attention.py`
- Shell entry: `scripts/build/flash_attention/build_flash_attention.sh`
- PowerShell entry: `scripts/build/flash_attention/build_flash_attention.ps1`
- Compatibility wrappers:
  - `scripts/build_flash_attention.py`
  - `scripts/build_flash_attention.sh`
  - `scripts/build_flash_attention.ps1`

## Prerequisites

- Windows dev drive with PowerShell or Linux/WSL with Bash.
- NVIDIA driver + CUDA toolkit available.
- On Windows with current Visual Studio 2022 toolchains, use CUDA `12.4+`.
- Active project environment (`uv sync` already run).
- Build tools (`cmake`, compiler toolchain).

## Recommended Command

```powershell
.\scripts\build_flash_attention.ps1
```

or

```powershell
uv run python scripts/build_flash_attention.py
```

The script rebuilds with:

```text
CMAKE_ARGS='-DGGML_CUDA=ON -DGGML_FLASH_ATTN=ON'
```

## Verify Import

```powershell
uv run python -c "from llama_cpp import Llama; print('ok')"
```

## Notes

- Performance impact varies by GPU, model, quantization, and context size.
- This document intentionally avoids fixed speedup claims.
- If you see `STL1002: Unexpected compiler version, expected CUDA 12.4 or newer`, upgrade the CUDA toolkit and reopen PowerShell.
- If Flash Attention fails, rebuild with CUDA only by adjusting `CMAKE_ARGS`.

## Related Files

- `scripts/build/flash_attention/build_flash_attention.py`
- `scripts/build/flash_attention/build_flash_attention.sh`
- `scripts/build/flash_attention/build_flash_attention.ps1`
- `core/gpu_utils.py`
