"""Compatibility wrapper for moved context-fetch script."""

from scripts.context.fetch_character_context import *  # noqa: F403
from scripts.context.fetch_character_context import main


if __name__ == "__main__":
    main()
