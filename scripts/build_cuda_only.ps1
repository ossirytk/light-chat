#!/usr/bin/env pwsh
[CmdletBinding()]
param()

$scriptPath = Join-Path $PSScriptRoot "build\flash_attention\build_flash_attention.ps1"
& $scriptPath -CudaOnly @args
exit $LASTEXITCODE
