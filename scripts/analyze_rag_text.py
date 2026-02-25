"""Compatibility wrapper for moved RAG analysis script."""

from scripts.rag.analyze_rag_text import *  # noqa: F403
from scripts.rag.analyze_rag_text import cli


if __name__ == "__main__":
    cli()
