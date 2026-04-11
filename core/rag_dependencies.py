from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

RAG_EXTRA_INSTALL_COMMAND = "uv sync --extra rag"

if TYPE_CHECKING:
    from types import ModuleType

    from chromadb.config import Settings
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_transformers import CrossEncoder


class MissingRagDependenciesError(RuntimeError):
    """Raised when optional RAG dependencies are unavailable."""


def _missing_rag_dependencies_error() -> MissingRagDependenciesError:
    return MissingRagDependenciesError(
        "RAG embedding dependencies are not installed. "
        f"Run `{RAG_EXTRA_INSTALL_COMMAND}` to enable vector retrieval and reranking."
    )


def import_vector_dependencies() -> tuple[ModuleType, type[Settings], type[Chroma], type[HuggingFaceEmbeddings]]:
    """Import optional vector-retrieval dependencies on demand."""
    try:
        chromadb_module = import_module("chromadb")
        chromadb_config_module = import_module("chromadb.config")
        langchain_chroma_module = import_module("langchain_chroma")
        huggingface_module = import_module("langchain_huggingface")
    except ImportError as exc:
        raise _missing_rag_dependencies_error() from exc

    return (
        chromadb_module,
        chromadb_config_module.Settings,
        langchain_chroma_module.Chroma,
        huggingface_module.HuggingFaceEmbeddings,
    )


def import_cross_encoder() -> type[CrossEncoder]:
    """Import the optional reranker dependency on demand."""
    try:
        sentence_transformers_module = import_module("sentence_transformers")
    except ImportError as exc:
        raise _missing_rag_dependencies_error() from exc

    return sentence_transformers_module.CrossEncoder
