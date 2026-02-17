#!/usr/bin/env python3
"""Script to rebuild llama-cpp-python with Flash Attention support using uv."""

import os
import shlex
import subprocess
import sys


def run_command(cmd: str, description: str = "") -> bool:
    """Run a shell command and return success status."""
    if description:
        print(f"\n{description}")  # noqa: T201
    print(f"$ {cmd}")  # noqa: T201
    try:
        result = subprocess.run(shlex.split(cmd), check=False)  # noqa: S603
    except Exception as e:
        print(f"Error: {e}")  # noqa: T201
        return False
    else:
        return result.returncode == 0


def main() -> int:  # noqa: PLR0912,PLR0915
    """Build llama-cpp-python with Flash Attention support."""
    print("=" * 60)  # noqa: T201
    print("Building llama-cpp-python with CUDA + Flash Attention")  # noqa: T201
    print("=" * 60)  # noqa: T201

    # Check if in virtual environment
    if not os.environ.get("VIRTUAL_ENV"):
        print("\n❌ Not in a Python virtual environment!")  # noqa: T201
        print("Please activate the uv environment first:")  # noqa: T201
        print("  source .venv/bin/activate")  # noqa: T201
        return 1

    venv_path = os.environ.get("VIRTUAL_ENV")
    print(f"✓ Virtual environment: {venv_path}")  # noqa: T201

    # Check system dependencies
    print("\nChecking system dependencies...")  # noqa: T201

    deps_ok = True
    if not run_command("which cmake > /dev/null", "Checking for cmake..."):
        print("  ❌ cmake not found - installing...")  # noqa: T201
        if not run_command("sudo apt-get update && sudo apt-get install -y cmake", "Installing cmake..."):
            print("  Failed to install cmake")  # noqa: T201
            deps_ok = False
        else:
            print("  ✓ cmake installed")  # noqa: T201
    else:
        print("  ✓ cmake available")  # noqa: T201

    if not run_command("which gcc > /dev/null", "Checking for gcc..."):
        print("  ❌ gcc not found - installing build-essential...")  # noqa: T201
        if not run_command("sudo apt-get install -y build-essential", "Installing build tools..."):
            print("  Failed to install build tools")  # noqa: T201
            deps_ok = False
        else:
            print("  ✓ build tools installed")  # noqa: T201
    else:
        print("  ✓ build tools available")  # noqa: T201

    if not run_command("which nvcc > /dev/null", "Checking for CUDA (nvcc)..."):
        print("  ⚠️  CUDA toolkit not found in PATH")  # noqa: T201
        print("  Install NVIDIA CUDA toolkit and add to PATH")  # noqa: T201
        print("  Or ensure CUDA is available via LD_LIBRARY_PATH")  # noqa: T201
    else:
        print("  ✓ CUDA available")  # noqa: T201

    if not deps_ok:
        print("\n❌ Required dependencies not available")  # noqa: T201
        return 1

    # Install build packages
    print("\n" + "=" * 60)  # noqa: T201
    print("Installing Python build dependencies with uv...")  # noqa: T201
    print("=" * 60)  # noqa: T201

    if not run_command("uv pip install cmake scikit-build-core", "Installing build tools..."):
        print("⚠️  Failed to install build tools via uv")  # noqa: T201

    # Build llama-cpp-python
    print("\n" + "=" * 60)  # noqa: T201
    print("Building llama-cpp-python with Flash Attention...")  # noqa: T201
    print("=" * 60)  # noqa: T201

    env_vars = "CMAKE_ARGS='-DGGML_CUDA=ON -DGGML_FLASH_ATTN=ON'"
    build_cmd = f"{env_vars} uv pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python"

    print("\nCMAKE_ARGS: -DGGML_CUDA=ON -DGGML_FLASH_ATTN=ON")  # noqa: T201
    print("(This will take several minutes - involves compilation)")  # noqa: T201
    print()  # noqa: T201

    if not run_command(build_cmd):
        return 1

    # Verify installation

    if run_command("uv run python -c \"from llama_cpp import Llama; print('✓ Successfully imported')\""):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
