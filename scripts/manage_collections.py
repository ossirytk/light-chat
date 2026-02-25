"""Compatibility wrapper for moved collection management script."""

from scripts.rag.manage_collections import *  # noqa: F403
from scripts.rag.manage_collections import cli


if __name__ == "__main__":
    cli()
