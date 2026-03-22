# Flash Attention Build Guide

Last verified: 2026-03-07

This repo includes build helpers for rebuilding `llama-cpp-python` with CUDA + Flash Attention.

For older NVIDIA GPUs that support CUDA but not Flash Attention, there is also a PowerShell CUDA-only helper.

## Build Helpers

- Python entry: `scripts/build/flash_attention/build_flash_attention.py`
- Shell entry: `scripts/build/flash_attention/build_flash_attention.sh`
- PowerShell entry: `scripts/build/flash_attention/build_flash_attention.ps1`
- PowerShell CUDA-only entry: `scripts/build_cuda_only.ps1`
- Compatibility wrappers:
  - `scripts/build_flash_attention.py`
  - `scripts/build_flash_attention.sh`
  - `scripts/build_flash_attention.ps1`
  - `scripts/build_cuda_only.ps1`

## Prerequisites

- Windows dev drive with PowerShell or Linux/WSL with Bash.
- NVIDIA driver + CUDA toolkit available.
- On Windows with current Visual Studio toolchains, use CUDA `12.4+`.
- For Pascal-era GPUs below compute capability `7.0` (for example Quadro P4000 / `6.1`), stay on CUDA `12.x`; CUDA `13.x` rejects `compute_61` during compiler detection.
- Flash Attention on NVIDIA requires a GPU with compute capability `7.0+` (Volta or newer).
- Active project environment (`uv sync` already run).
- Build tools (`cmake`, compiler toolchain).

## Recommended Command

```powershell
.\scripts\build_flash_attention.ps1
```

On Windows, the PowerShell helper now tries to auto-load the Visual Studio x64 C++ toolchain into the current session when `cl.exe` is missing from `PATH`, and it normalizes CUDA environment variables to the active toolkit before building.

When multiple CUDA toolkits are installed, the helper prefers `CUDACXX`, then `CUDA_PATH`, then `CUDAToolkit_ROOT`, and only falls back to `nvcc` from `PATH` if none of those are set to a valid toolkit.

For Windows CUDA builds, the helper also forces the `Ninja` generator and passes the selected `nvcc.exe` path directly to CMake so it does not depend on Visual Studio's CUDA toolset integration.

For CUDA-only rebuilds on older GPUs, run:

```powershell
.\scripts\build_cuda_only.ps1
```

or

```powershell
uv run python scripts/build_flash_attention.py
```

The script rebuilds with:

```text
CMAKE_ARGS='-DGGML_CUDA=ON -DGGML_CUDA_FA=ON'
```

The CUDA-only PowerShell helper rebuilds with:

```text
CMAKE_ARGS='-DGGML_CUDA=ON -DGGML_CUDA_FA=OFF'
```

## Verify Import

```powershell
uv run python -c "from llama_cpp import Llama; print('ok')"
```

## Notes

- Performance impact varies by GPU, model, quantization, and context size.
- This document intentionally avoids fixed speedup claims.
- If you see `STL1002: Unexpected compiler version, expected CUDA 12.4 or newer`, upgrade the CUDA toolkit and reopen PowerShell.
- If the script reports a compute capability below `7.0`, your GPU is too old for Flash Attention; rebuild with CUDA only by setting `GGML_CUDA_FA=OFF`.
- If `build_cuda_only.ps1` fails with `nvcc fatal : Unsupported gpu architecture 'compute_61'`, your installed CUDA toolkit is too new for Pascal/`6.x` GPUs. Install CUDA `12.x`, reopen PowerShell, and retry.
- On Visual Studio `18` / 2026, the PowerShell helper automatically adds `--allow-unsupported-compiler` for `nvcc` so CUDA `12.x` can build with the newer MSVC toolchain.
- If Flash Attention fails, rebuild with CUDA only by adjusting `CMAKE_ARGS`.

## Related Files

- `scripts/build/flash_attention/build_flash_attention.py`
- `scripts/build/flash_attention/build_flash_attention.sh`
- `scripts/build/flash_attention/build_flash_attention.ps1`
- `scripts/build_cuda_only.ps1`
- `core/gpu_utils.py`
