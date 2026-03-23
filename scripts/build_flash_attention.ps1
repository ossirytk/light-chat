#!/usr/bin/env pwsh
[CmdletBinding()]
param()

$scriptPath = Join-Path $PSScriptRoot "build\flash_attention\build_flash_attention.ps1"
& $scriptPath @args
exit $LASTEXITCODE
