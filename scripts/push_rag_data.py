"""Compatibility wrapper for moved RAG push script."""

from scripts.rag.push_rag_data import *  # noqa: F403
from scripts.rag.push_rag_data import main


if __name__ == "__main__":
    main()
