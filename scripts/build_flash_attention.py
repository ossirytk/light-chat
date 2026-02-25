"""Compatibility wrapper for moved Flash Attention build script."""

from scripts.build.flash_attention.build_flash_attention import *  # noqa: F403
from scripts.build.flash_attention.build_flash_attention import main


if __name__ == "__main__":
    raise SystemExit(main())
