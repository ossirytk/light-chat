"""Compatibility wrapper for moved legacy RAG preparation script."""

from scripts.rag.old_prepare_rag import *  # noqa: F403
from scripts.rag.old_prepare_rag import main


if __name__ == "__main__":
    main()
