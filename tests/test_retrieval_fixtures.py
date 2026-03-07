"""Evaluation fixtures for retrieval quality regression checks.

The structure test always runs. Live retrieval checks are opt-in because they
require an existing local ChromaDB collection and embedding model.
"""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import chromadb
from chromadb.config import Settings

from core.config import load_runtime_config
from core.conversation_manager import ConversationManager

FIXTURES_PATH = Path("tests/fixtures/retrieval_fixtures.json")
HARD_FIXTURES_PATH = Path("tests/fixtures/retrieval_fixtures_hard.json")
NEGATIVE_FIXTURES_PATH = Path("tests/fixtures/retrieval_fixtures_negative.json")
RERANK_FIXTURES_PATH = Path("tests/fixtures/retrieval_fixtures_rerank.json")


def _make_eval_manager() -> ConversationManager:
    config = load_runtime_config().flat
    mgr = object.__new__(ConversationManager)
    mgr.configs = config
    mgr.rag_k = int(config.get("RAG_K", 3))
    mgr.rag_k_mes = int(config.get("RAG_K_MES", mgr.rag_k))
    mgr.persist_directory = str(config.get("PERSIST_DIRECTORY", "./character_storage/"))
    mgr.embedding_cache = str(config.get("EMBEDDING_CACHE", "./embedding_models/"))
    mgr.runtime_config = SimpleNamespace(
        use_mmr=bool(config.get("USE_MMR", True)),
        rag_fetch_k=int(config.get("RAG_FETCH_K", 20)),
        lambda_mult=float(config.get("LAMBDA_MULT", 0.75)),
        rag_rerank_enabled=bool(config.get("RAG_RERANK_ENABLED", False)),
        rag_rerank_top_n=int(config.get("RAG_RERANK_TOP_N", 8)),
        rag_rerank_model=str(config.get("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
        rag_telemetry_enabled=bool(config.get("RAG_TELEMETRY_ENABLED", False)),
        rag_multi_query_enabled=bool(config.get("RAG_MULTI_QUERY_ENABLED", True)),
        rag_multi_query_max_variants=int(config.get("RAG_MULTI_QUERY_MAX_VARIANTS", 3)),
        rag_sentence_compression_enabled=bool(config.get("RAG_SENTENCE_COMPRESSION_ENABLED", True)),
        rag_sentence_compression_max_sentences=int(config.get("RAG_SENTENCE_COMPRESSION_MAX_SENTENCES", 8)),
    )
    mgr._vector_client = None  # noqa: SLF001
    mgr._vector_embedder = None  # noqa: SLF001
    mgr._cross_encoder = None  # noqa: SLF001
    mgr._vector_dbs = {}  # noqa: SLF001
    return mgr


class TestRetrievalFixtures(unittest.TestCase):
    """Fixture-driven checks for retrieval quality expectations."""

    def _assert_fixture_schema(self, fixture_path: Path) -> None:
        data = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertIn("collection", data)
        self.assertIn("cases", data)
        self.assertIsInstance(data["cases"], list)
        self.assertGreater(len(data["cases"]), 0)
        if "dashboard_ks" in data:
            self.assertIsInstance(data["dashboard_ks"], list)
            self.assertGreater(len(data["dashboard_ks"]), 0)
            self.assertTrue(all(isinstance(value, int) and value > 0 for value in data["dashboard_ks"]))

        for case in data["cases"]:
            self.assertIn("id", case)
            self.assertIn("query", case)
            self.assertIn("expected_snippets", case)
            self.assertIsInstance(case["expected_snippets"], list)
            self.assertGreater(len(case["expected_snippets"]), 0)
            if "forbidden_snippets" in case:
                self.assertIsInstance(case["forbidden_snippets"], list)
            if "min_expected_matches" in case:
                self.assertIsInstance(case["min_expected_matches"], int)
                self.assertGreaterEqual(case["min_expected_matches"], 1)

    def test_fixture_schema_is_valid(self) -> None:
        self._assert_fixture_schema(FIXTURES_PATH)
        self._assert_fixture_schema(HARD_FIXTURES_PATH)
        self._assert_fixture_schema(NEGATIVE_FIXTURES_PATH)
        self._assert_fixture_schema(RERANK_FIXTURES_PATH)

    @unittest.skipUnless(os.getenv("RUN_RAG_FIXTURES") == "1", "Set RUN_RAG_FIXTURES=1 to run live retrieval checks")
    def test_live_retrieval_matches_expected_snippets(self) -> None:
        data = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
        default_collection = str(data["collection"])
        default_k = int(data.get("k", 5))

        manager = _make_eval_manager()
        client = chromadb.PersistentClient(
            path=manager.persist_directory, settings=Settings(anonymized_telemetry=False)
        )
        available_collections = {collection.name for collection in client.list_collections()}

        for case in data["cases"]:
            collection_name = str(case.get("collection", default_collection))
            if collection_name not in available_collections:
                self.skipTest(f"Missing collection '{collection_name}' required by fixture '{case['id']}'")

            chunks = manager._search_collection(collection_name, str(case["query"]), [None], k=default_k)  # noqa: SLF001
            merged = "\n".join(chunks)
            expected = [snippet for snippet in case["expected_snippets"] if isinstance(snippet, str) and snippet]
            self.assertTrue(expected, "Fixture must define at least one expected snippet")
            self.assertTrue(
                any(snippet in merged for snippet in expected),
                f"Fixture '{case['id']}' did not return any expected snippet in top-{default_k} results",
            )


if __name__ == "__main__":
    unittest.main()
