#!/usr/bin/env bash
# Compatibility wrapper for moved Flash Attention build shell script.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/build/flash_attention/build_flash_attention.sh" "$@"
