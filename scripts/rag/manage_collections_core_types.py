"""Shared types and data models for collection management core."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

type KeyItem = dict[str, object]
type KeyMatch = dict[str, str]
type WhereFilter = dict[str, object] | None

EMBEDDING_MODEL_METADATA_KEY = "embedding:model"
EMBEDDING_DIMENSION_METADATA_KEY = "embedding:dimension"
EMBEDDING_NORMALIZE_METADATA_KEY = "embedding:normalize"


@dataclass
class ManagementContext:
    client: chromadb.PersistentClient
    persist_directory: str
    embedder: HuggingFaceEmbeddings
    key_storage: str


@dataclass
class FixtureEvalContext:
    default_collection: str
    default_k: int
    evaluation_k: int
    dashboard_ks: list[int]
    available_collections: set[str]
    db_cache: dict[str, Chroma]
    retrieval_mode: str
    runtime_manager: object | None
    show_failures: bool


@dataclass
class FixtureCaseResult:
    case_id: str
    rank: int | None
    status: str
    query: str
    collection: str
    expected_snippets: list[str]
    min_expected_matches: int = 1
    expected_total: int = 0
    matched_expected: int = 0
    expected_recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    average_precision_at_k: float = 0.0
    forbidden_snippets: list[str] = field(default_factory=list)
    forbidden_matches: list[str] = field(default_factory=list)
    forbidden_hit: bool = False


@dataclass
class FixtureEvalOptions:
    fixture_file: Path
    k: int | None
    retrieval_mode: str
    persist_directory: str | None
    embedding_model: str | None
    embedding_device: str | None
    show_failures: bool


@dataclass
class FixtureEvalRun:
    default_k: int
    case_results: list[FixtureCaseResult]
    skipped: int
    metrics: dict[str, float]
    report: dict[str, Any]


__all__ = [
    "EMBEDDING_DIMENSION_METADATA_KEY",
    "EMBEDDING_MODEL_METADATA_KEY",
    "EMBEDDING_NORMALIZE_METADATA_KEY",
    "FixtureCaseResult",
    "FixtureEvalContext",
    "FixtureEvalOptions",
    "FixtureEvalRun",
    "KeyItem",
    "KeyMatch",
    "ManagementContext",
    "WhereFilter",
]
