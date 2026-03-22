#!/usr/bin/env pwsh
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Message)

    $line = "=" * 58
    Write-Host ""
    Write-Host $line
    Write-Host $Message
    Write-Host $line
}

function Test-CommandAvailable {
    param([string]$Name)

    return $null -ne (Get-Command -Name $Name -ErrorAction SilentlyContinue)
}

function Get-CommandOutput {
    param(
        [string]$Name,
        [string[]]$Arguments = @()
    )

    $command = Get-Command -Name $Name -ErrorAction SilentlyContinue
    if (-not $command) {
        return $null
    }

    $output = & $command.Source @Arguments 2>&1
    return ($output | Out-String).Trim()
}

function Get-CudaVersion {
    $nvccOutput = Get-CommandOutput -Name "nvcc" -Arguments @("--version")
    if (-not $nvccOutput) {
        return $null
    }

    $match = [regex]::Match($nvccOutput, "release\s+(?<version>\d+\.\d+)")
    if (-not $match.Success) {
        return $null
    }

    return [version]$match.Groups["version"].Value
}

function Invoke-Step {
    param(
        [string]$Description,
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host $Description
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed during: $Description"
    }
}

Write-Section "Building llama-cpp-python with CUDA + Flash Attention"

if (-not $env:VIRTUAL_ENV) {
    Write-Host "❌ Not in a Python virtual environment!"
    Write-Host "Please activate the project environment first:"
    Write-Host "  .venv\Scripts\Activate.ps1"
    exit 1
}

Write-Host "✓ Virtual environment: $($env:VIRTUAL_ENV)"

Write-Host ""
Write-Host "Checking build dependencies..."

if (-not (Test-CommandAvailable "uv")) {
    Write-Host "❌ uv not found in PATH."
    Write-Host "Install uv and ensure it is available from PowerShell."
    exit 1
}

if (-not (Test-CommandAvailable "python")) {
    Write-Host "❌ python not found in PATH."
    exit 1
}

if (-not (Test-CommandAvailable "cmake")) {
    Write-Host "❌ cmake not found in PATH."
    Write-Host "Install CMake and reopen PowerShell so the PATH update is picked up."
    exit 1
}

if (-not (Test-CommandAvailable "nvcc")) {
    Write-Host "⚠️  nvcc not found in PATH."
    Write-Host "CUDA toolkit may still be installed elsewhere, but the build expects CUDA tooling to be available."
}
else {
    Write-Host "✓ CUDA toolkit detected"

    $cudaVersion = Get-CudaVersion
    if ($null -eq $cudaVersion) {
        Write-Host "⚠️  Could not determine the CUDA toolkit version from nvcc."
    }
    else {
        Write-Host "✓ CUDA version: $cudaVersion"
        $minimumCudaVersion = [version]"12.4"
        if ($cudaVersion -lt $minimumCudaVersion) {
            Write-Host "❌ CUDA $cudaVersion is too old for this Windows/MSVC build."
            Write-Host "Current MSVC headers require CUDA 12.4 or newer and fail with STL1002 during CMake compiler detection."
            Write-Host "Upgrade the NVIDIA CUDA toolkit to 12.4+ and reopen PowerShell before retrying."
            exit 1
        }
    }
}

if (-not (Test-CommandAvailable "cl")) {
    Write-Host "⚠️  Visual C++ Build Tools (cl.exe) not found in PATH."
    Write-Host "If the build fails, open a Developer PowerShell or install the MSVC build tools."
}
else {
    Write-Host "✓ MSVC build tools detected"
}

Write-Host "✓ Required commands available"

Invoke-Step "Installing Python build tools with uv..." {
    uv pip install cmake scikit-build-core
}

Write-Section "Building llama-cpp-python with Flash Attention"
Write-Host ""
Write-Host "CMAKE_ARGS: -DGGML_CUDA=ON -DGGML_FLASH_ATTN=ON"
Write-Host ""

$previousCmakeArgs = $env:CMAKE_ARGS
$env:CMAKE_ARGS = "-DGGML_CUDA=ON -DGGML_FLASH_ATTN=ON"

try {
    Invoke-Step "Reinstalling llama-cpp-python..." {
        uv pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
    }

    Write-Section "Build Complete"

    Invoke-Step "Testing installation..." {
        uv run python -c "from llama_cpp import Llama; print('✓ llama-cpp-python imported successfully')"
    }
}
finally {
    $env:CMAKE_ARGS = $previousCmakeArgs
}

Write-Host ""
Write-Host "To verify Flash Attention support, run:"
Write-Host '  uv run python -c "from llama_cpp import __version__; print(__version__)"'
