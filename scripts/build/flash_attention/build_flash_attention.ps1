#!/usr/bin/env pwsh
[CmdletBinding()]
param(
    [switch]$CudaOnly
)

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

function Resolve-CudaToolkitCandidate {
    param([string]$Candidate)

    if (-not $Candidate) {
        return $null
    }

    $normalizedCandidate = $Candidate.Trim().Trim('"').TrimEnd('\')
    if (-not $normalizedCandidate) {
        return $null
    }

    if ((Split-Path -Leaf $normalizedCandidate) -ieq "nvcc.exe") {
        $normalizedCandidate = Split-Path -Parent (Split-Path -Parent $normalizedCandidate)
    }
    elseif ((Split-Path -Leaf $normalizedCandidate) -ieq "bin") {
        $normalizedCandidate = Split-Path -Parent $normalizedCandidate
    }

    if (Test-Path (Join-Path $normalizedCandidate "include")) {
        return $normalizedCandidate
    }

    return $null
}

function Get-CudaVersion {
    param([string]$ToolkitRoot)

    $nvccPath = $null
    if ($ToolkitRoot) {
        $candidateNvccPath = Join-Path $ToolkitRoot "bin\nvcc.exe"
        if (Test-Path $candidateNvccPath) {
            $nvccPath = $candidateNvccPath
        }
    }

    if (-not $nvccPath) {
        $nvccCommand = Get-Command -Name "nvcc" -ErrorAction SilentlyContinue
        if (-not $nvccCommand) {
            return $null
        }

        $nvccPath = $nvccCommand.Source
    }

    $nvccOutput = & $nvccPath --version 2>&1 | Out-String
    if (-not $nvccOutput) {
        return $null
    }

    $match = [regex]::Match($nvccOutput, "release\s+(?<version>\d+\.\d+)")
    if (-not $match.Success) {
        return $null
    }

    return [version]$match.Groups["version"].Value
}

function Get-CudaToolkitRoot {
    foreach ($candidate in @($env:CUDACXX, $env:CUDA_PATH, $env:CUDAToolkit_ROOT)) {
        $resolvedToolkitRoot = Resolve-CudaToolkitCandidate -Candidate $candidate
        if ($resolvedToolkitRoot) {
            return $resolvedToolkitRoot
        }
    }

    $nvccCommand = Get-Command -Name "nvcc" -ErrorAction SilentlyContinue
    if ($nvccCommand) {
        $resolvedToolkitRoot = Resolve-CudaToolkitCandidate -Candidate $nvccCommand.Source
        if ($resolvedToolkitRoot) {
            return $resolvedToolkitRoot
        }
    }

    return $null
}

function Normalize-CudaEnvironment {
    $toolkitRoot = Get-CudaToolkitRoot
    if (-not $toolkitRoot) {
        return $null
    }

    $nvccPath = Join-Path $toolkitRoot "bin\nvcc.exe"
    $didNormalize = $false

    foreach ($entry in @(
            @{ Name = "CUDA_PATH"; Value = $toolkitRoot },
            @{ Name = "CUDAToolkit_ROOT"; Value = $toolkitRoot },
            @{ Name = "CUDACXX"; Value = $nvccPath }
        )) {
        if ($entry.Name -eq "CUDACXX" -and -not (Test-Path $entry.Value)) {
            continue
        }

        $currentValue = [Environment]::GetEnvironmentVariable($entry.Name, "Process")
        if ($currentValue -ne $entry.Value) {
            [Environment]::SetEnvironmentVariable($entry.Name, $entry.Value, "Process")
            $didNormalize = $true
        }
    }

    $cudaBinPath = Join-Path $toolkitRoot "bin"
    $pathEntries = @(
        [Environment]::GetEnvironmentVariable("PATH", "Process") -split ';' |
            Where-Object { $_ }
    )
    $filteredPathEntries = @(
        $pathEntries | Where-Object {
            $_.TrimEnd('\') -notmatch '\\NVIDIA GPU Computing Toolkit\\CUDA\\v[^\\]+\\bin$'
        }
    )
    $normalizedPathEntries = @($cudaBinPath) + $filteredPathEntries
    $normalizedPath = ($normalizedPathEntries | Select-Object -Unique) -join ';'
    if ([Environment]::GetEnvironmentVariable("PATH", "Process") -ne $normalizedPath) {
        [Environment]::SetEnvironmentVariable("PATH", $normalizedPath, "Process")
        $didNormalize = $true
    }

    if ($didNormalize) {
        Write-Host "✓ Normalized CUDA toolkit environment to: $toolkitRoot"
    }

    return $toolkitRoot
}

function Get-NvidiaGpuInfo {
    $queryOutput = Get-CommandOutput -Name "nvidia-smi" -Arguments @("--query-gpu=name,compute_cap", "--format=csv,noheader")
    if (-not $queryOutput) {
        return @()
    }

    return @(
        $queryOutput -split "\r?\n" |
            Where-Object { $_.Trim() } |
            ForEach-Object {
                $fields = $_ -split ",", 2
                if ($fields.Count -lt 2) {
                    return
                }

                [PSCustomObject]@{
                    Name = $fields[0].Trim()
                    ComputeCapability = $fields[1].Trim()
                }
            }
    )
}

function Get-VsWherePath {
    $candidatePaths = @(
        (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"),
        (Join-Path $env:ProgramFiles "Microsoft Visual Studio\Installer\vswhere.exe")
    )

    foreach ($candidatePath in $candidatePaths) {
        if ($candidatePath -and (Test-Path $candidatePath)) {
            return $candidatePath
        }
    }

    return $null
}

function Get-VisualStudioInstallationPath {
    $vswherePath = Get-VsWherePath
    if ($vswherePath) {
        $installationPath = (& $vswherePath -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null | Select-Object -First 1).Trim()
        if ($installationPath) {
            return $installationPath
        }
    }

    $candidateInstallations = @()
    foreach ($year in @("2026", "2022")) {
        foreach ($edition in @("Community", "Professional", "Enterprise", "BuildTools")) {
            $candidateInstallations += Join-Path $env:ProgramFiles "Microsoft Visual Studio\$year\$edition"
        }
    }

    foreach ($candidateInstallation in $candidateInstallations) {
        if (Test-Path (Join-Path $candidateInstallation "VC\Tools\MSVC")) {
            return $candidateInstallation
        }
    }

    return $null
}

function Import-EnvironmentVariablesFromOutput {
    param([string[]]$Lines)

    foreach ($line in $Lines) {
        if ($line -notmatch "^(?<name>[^=]+)=(?<value>.*)$") {
            continue
        }

        [Environment]::SetEnvironmentVariable($matches["name"], $matches["value"], "Process")
    }
}

function Import-VsDevEnvironment {
    param([string]$InstallationPath)

    $vsDevCmdPath = Join-Path $InstallationPath "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path $vsDevCmdPath)) {
        return $false
    }

    $previousTelemetryPreference = $env:VSCMD_SKIP_SENDTELEMETRY
    $env:VSCMD_SKIP_SENDTELEMETRY = "1"

    try {
        $commandOutput = & cmd.exe /d /s /c "`"$vsDevCmdPath`" -host_arch=x64 -arch=x64 >nul && set" 2>&1
        if ($LASTEXITCODE -ne 0) {
            return $false
        }

        Import-EnvironmentVariablesFromOutput -Lines $commandOutput
        return $true
    }
    finally {
        $env:VSCMD_SKIP_SENDTELEMETRY = $previousTelemetryPreference
    }
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
        throw "Command failed during: $Description (exit code $LASTEXITCODE)"
    }
}

function Ensure-MsvcBuildToolsAvailable {
    if (Test-CommandAvailable "cl") {
        Write-Host "✓ MSVC build tools detected"
        return
    }

    Write-Host "⚠️  Visual C++ Build Tools (cl.exe) not found in PATH."
    Write-Host "Attempting to load the Visual Studio C++ toolchain for this PowerShell session..."

    $installationPath = Get-VisualStudioInstallationPath
    if (-not $installationPath) {
        Write-Host "❌ Could not locate a Visual Studio installation with C++ tools."
        Write-Host "Install the Desktop development with C++ workload or open Developer PowerShell manually."
        exit 1
    }

    if (-not (Import-VsDevEnvironment -InstallationPath $installationPath)) {
        Write-Host "❌ Found Visual Studio at: $installationPath"
        Write-Host "Failed to import the MSVC developer environment into this PowerShell session."
        Write-Host "Open Developer PowerShell for Visual Studio and retry the build."
        exit 1
    }

    if (-not (Test-CommandAvailable "cl")) {
        Write-Host "❌ Loaded Visual Studio environment from: $installationPath"
        Write-Host "cl.exe is still unavailable in PATH after importing the developer environment."
        Write-Host "Open Developer PowerShell for Visual Studio and retry the build."
        exit 1
    }

    Write-Host "✓ Loaded MSVC build tools from: $installationPath"
}

function Ensure-FlashAttentionSupported {
    $minimumComputeCapability = [version]"7.0"
    $gpuInfo = @(Get-NvidiaGpuInfo)

    if ($gpuInfo.Count -eq 0) {
        Write-Host "⚠️  Could not determine NVIDIA GPU compute capability from nvidia-smi."
        Write-Host "Continuing, but Flash Attention may fail later if the GPU is too old."
        return
    }

    $unsupportedGpus = @(
        $gpuInfo | Where-Object {
            [version]$_.ComputeCapability -lt $minimumComputeCapability
        }
    )

    if ($unsupportedGpus.Count -eq 0) {
        Write-Host "✓ NVIDIA GPU compute capability supports Flash Attention"
        return
    }

    Write-Host "❌ Flash Attention requires an NVIDIA GPU with compute capability 7.0 or newer (Volta+)."
    foreach ($gpu in $unsupportedGpus) {
        Write-Host "Detected GPU: $($gpu.Name) (compute capability $($gpu.ComputeCapability))"
    }
    Write-Host "This GPU can still use the CUDA backend, but llama.cpp Flash Attention kernels do not build for it."
    Write-Host "Rebuild without Flash Attention by using CMAKE_ARGS='-DGGML_CUDA=ON -DGGML_CUDA_FA=OFF'."
    exit 1
}

function Ensure-CudaToolkitSupportsDetectedGpus {
    param([version]$CudaVersion)

    if ($null -eq $CudaVersion -or $CudaVersion.Major -lt 13) {
        return
    }

    $gpuInfo = @(Get-NvidiaGpuInfo)
    if ($gpuInfo.Count -eq 0) {
        Write-Host "⚠️  Could not determine NVIDIA GPU compute capability from nvidia-smi."
        Write-Host "Continuing, but CUDA compiler compatibility could not be verified."
        return
    }

    $unsupportedGpus = @(
        $gpuInfo | Where-Object {
            [version]$_.ComputeCapability -lt [version]"7.0"
        }
    )

    if ($unsupportedGpus.Count -eq 0) {
        return
    }

    Write-Host "❌ CUDA $CudaVersion cannot build this project for NVIDIA GPUs below compute capability 7.0 on Windows."
    foreach ($gpu in $unsupportedGpus) {
        Write-Host "Detected GPU: $($gpu.Name) (compute capability $($gpu.ComputeCapability))"
    }
    Write-Host "The current failure comes from nvcc rejecting the target architecture during compiler detection."
    Write-Host "For Pascal-era GPUs such as compute capability 6.1, CUDA 13.x fails with: nvcc fatal : Unsupported gpu architecture 'compute_61'."
    Write-Host "Install a CUDA 12.x toolkit (12.4+ recommended), reopen PowerShell, and rerun .\scripts\build_cuda_only.ps1."
    exit 1
}

function Get-VisualStudioMajorVersion {
    $visualStudioVersion = [Environment]::GetEnvironmentVariable("VisualStudioVersion", "Process")
    if (-not $visualStudioVersion) {
        return $null
    }

    $parsedVersion = $null
    if (-not [version]::TryParse($visualStudioVersion, [ref]$parsedVersion)) {
        return $null
    }

    return $parsedVersion.Major
}

$enableFlashAttention = -not $CudaOnly
$buildDisplayName = if ($enableFlashAttention) {
    "Building llama-cpp-python with CUDA + Flash Attention"
}
else {
    "Building llama-cpp-python with CUDA only"
}
$cmakeArgs = if ($enableFlashAttention) {
    "-DGGML_CUDA=ON -DGGML_CUDA_FA=ON"
}
else {
    "-DGGML_CUDA=ON -DGGML_CUDA_FA=OFF"
}
$toolkitRoot = $null

Write-Section $buildDisplayName

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

    $toolkitRoot = Normalize-CudaEnvironment
    $cudaVersion = Get-CudaVersion -ToolkitRoot $toolkitRoot
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
    if ($enableFlashAttention) {
        Ensure-FlashAttentionSupported
    }
    else {
        Ensure-CudaToolkitSupportsDetectedGpus -CudaVersion $cudaVersion
    }
}

Ensure-MsvcBuildToolsAvailable

Write-Host "✓ Required commands available"

Invoke-Step "Installing Python build tools with uv..." {
    uv pip install cmake ninja scikit-build-core
}

Write-Section $buildDisplayName
Write-Host ""
$effectiveCmakeArgs = @($cmakeArgs)
if ($toolkitRoot) {
    $effectiveCmakeArgs += "-DCUDAToolkit_ROOT=""$toolkitRoot"""
    $effectiveCmakeArgs += "-DCMAKE_CUDA_COMPILER=""" + (Join-Path $toolkitRoot "bin\nvcc.exe") + """"
}
if ($IsWindows) {
    $visualStudioMajorVersion = Get-VisualStudioMajorVersion
    if ($visualStudioMajorVersion -ge 18) {
        $effectiveCmakeArgs += "-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler"
    }
}

$effectiveCmakeArgsString = $effectiveCmakeArgs -join " "

Write-Host "CMAKE_ARGS: $effectiveCmakeArgsString"
if ($IsWindows) {
    Write-Host "CMAKE_GENERATOR: Ninja"
}
Write-Host ""

$previousCmakeArgs = $env:CMAKE_ARGS
$previousCmakeGenerator = $env:CMAKE_GENERATOR
$env:CMAKE_ARGS = $effectiveCmakeArgsString
if ($IsWindows) {
    $env:CMAKE_GENERATOR = "Ninja"
}

try {
    Invoke-Step "Reinstalling llama-cpp-python..." {
        uv pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
    }

    Write-Section "Build Complete"

    Invoke-Step "Testing installation..." {
        uv run python -c "from llama_cpp import Llama; print('llama-cpp-python imported successfully')"
    }
}
finally {
    $env:CMAKE_ARGS = $previousCmakeArgs
    $env:CMAKE_GENERATOR = $previousCmakeGenerator
}

Write-Host ""
if ($enableFlashAttention) {
    Write-Host "To verify Flash Attention support, run:"
}
else {
    Write-Host "To verify the CUDA-only build, run:"
}
Write-Host '  uv run python -c "from llama_cpp import __version__; print(__version__)"'
