"""Unit tests for ConversationManager._search_collection retrieval logic."""

import unittest
from unittest.mock import MagicMock

from langchain_core.documents.base import Document

from core.conversation_manager import ConversationManager


def _make_manager(config_overrides: dict | None = None) -> ConversationManager:
    """Create a minimal ConversationManager-like object without loading real models."""
    mgr = object.__new__(ConversationManager)
    base_config: dict = {
        "RAG_K": 5,
        "RAG_K_MES": 3,
        "RAG_FETCH_K": 20,
        "LAMBDA_MULT": 0.75,
        "USE_MMR": True,
    }
    if config_overrides:
        base_config.update(config_overrides)
    mgr.configs = base_config
    mgr.rag_k = int(base_config["RAG_K"])
    mgr.rag_k_mes = int(base_config["RAG_K_MES"])
    mgr._vector_dbs = {}  # noqa: SLF001
    return mgr


def _make_docs(*contents: str) -> list[Document]:
    return [Document(page_content=c) for c in contents]


def _make_scored_docs(*contents: str, score: float = 0.5) -> list[tuple[Document, float]]:
    return [(Document(page_content=c), score) for c in contents]


class TestSearchCollectionMMR(unittest.TestCase):
    """Validate _search_collection MMR path."""

    def _get_mock_db(self, mgr: ConversationManager, results: list[Document]) -> MagicMock:
        db = MagicMock()
        db.max_marginal_relevance_search.return_value = results
        mgr._vector_dbs["col"] = db  # noqa: SLF001
        return db

    def test_mmr_with_filter_calls_mmr_search(self) -> None:
        mgr = _make_manager({"USE_MMR": True})
        db = self._get_mock_db(mgr, _make_docs("chunk A", "chunk B"))
        result = mgr._search_collection("col", "query", [{"uuid": "x"}])  # noqa: SLF001
        db.max_marginal_relevance_search.assert_called_once()
        call_kwargs = db.max_marginal_relevance_search.call_args.kwargs
        self.assertIn("filter", call_kwargs)
        self.assertEqual(call_kwargs["filter"], {"uuid": "x"})
        self.assertEqual(result, ["chunk A", "chunk B"])

    def test_mmr_with_none_filter_omits_filter_kwarg(self) -> None:
        """When where is None, filter must not be passed to the search method."""
        mgr = _make_manager({"USE_MMR": True})
        db = self._get_mock_db(mgr, _make_docs("chunk A"))
        result = mgr._search_collection("col", "query", [None])  # noqa: SLF001
        db.max_marginal_relevance_search.assert_called_once()
        call_kwargs = db.max_marginal_relevance_search.call_args.kwargs
        self.assertNotIn("filter", call_kwargs)
        self.assertEqual(result, ["chunk A"])

    def test_mmr_fallback_to_unfiltered_when_filtered_empty(self) -> None:
        """MMR should try the next filter (None = unfiltered) when a filtered search returns nothing."""
        mgr = _make_manager({"USE_MMR": True})
        db = MagicMock()
        # First call (filtered) returns nothing; second call (unfiltered) returns a result
        db.max_marginal_relevance_search.side_effect = [[], _make_docs("fallback chunk")]
        mgr._vector_dbs["col"] = db  # noqa: SLF001
        result = mgr._search_collection("col", "query", [{"uuid": "x"}, None])  # noqa: SLF001
        self.assertEqual(db.max_marginal_relevance_search.call_count, 2)
        self.assertEqual(result, ["fallback chunk"])

    def test_mmr_fetch_k_floored_to_k(self) -> None:
        """fetch_k passed to MMR must be at least as large as k."""
        mgr = _make_manager({"USE_MMR": True, "RAG_FETCH_K": 2})
        db = self._get_mock_db(mgr, _make_docs("x"))
        mgr._search_collection("col", "query", [None], k=10)  # noqa: SLF001
        call_kwargs = db.max_marginal_relevance_search.call_args.kwargs
        self.assertGreaterEqual(call_kwargs["fetch_k"], 10)


class TestSearchCollectionSimilarity(unittest.TestCase):
    """Validate _search_collection non-MMR (similarity + threshold) path."""

    def _get_mock_db(self, mgr: ConversationManager, scored_results: list) -> MagicMock:
        db = MagicMock()
        db.similarity_search_with_score.return_value = scored_results
        mgr._vector_dbs["col"] = db  # noqa: SLF001
        return db

    def test_similarity_with_filter_passes_filter_kwarg(self) -> None:
        mgr = _make_manager({"USE_MMR": False})
        db = self._get_mock_db(mgr, _make_scored_docs("chunk A", score=0.3))
        result = mgr._search_collection("col", "query", [{"uuid": "y"}])  # noqa: SLF001
        call_kwargs = db.similarity_search_with_score.call_args.kwargs
        self.assertIn("filter", call_kwargs)
        self.assertEqual(result, ["chunk A"])

    def test_similarity_with_none_filter_omits_filter_kwarg(self) -> None:
        mgr = _make_manager({"USE_MMR": False})
        db = self._get_mock_db(mgr, _make_scored_docs("chunk A", score=0.3))
        result = mgr._search_collection("col", "query", [None])  # noqa: SLF001
        call_kwargs = db.similarity_search_with_score.call_args.kwargs
        self.assertNotIn("filter", call_kwargs)
        self.assertEqual(result, ["chunk A"])

    def test_score_threshold_filters_high_distance_chunks(self) -> None:
        mgr = _make_manager({"USE_MMR": False, "RAG_SCORE_THRESHOLD": 1.0})
        scored = [
            (Document(page_content="close chunk"), 0.4),
            (Document(page_content="far chunk"), 1.5),
        ]
        db = MagicMock()
        db.similarity_search_with_score.return_value = scored
        mgr._vector_dbs["col"] = db  # noqa: SLF001
        result = mgr._search_collection("col", "query", [None])  # noqa: SLF001
        self.assertEqual(result, ["close chunk"])
        self.assertNotIn("far chunk", result)

    def test_score_threshold_none_returns_all_chunks(self) -> None:
        """When RAG_SCORE_THRESHOLD is absent, all chunks are returned regardless of score."""
        mgr = _make_manager({"USE_MMR": False})
        scored = [
            (Document(page_content="close"), 0.2),
            (Document(page_content="far"), 9.9),
        ]
        db = MagicMock()
        db.similarity_search_with_score.return_value = scored
        mgr._vector_dbs["col"] = db  # noqa: SLF001
        result = mgr._search_collection("col", "query", [None])  # noqa: SLF001
        self.assertEqual(result, ["close", "far"])

    def test_empty_query_returns_empty(self) -> None:
        mgr = _make_manager({"USE_MMR": False})
        result = mgr._search_collection("col", "", [None])  # noqa: SLF001
        self.assertEqual(result, [])

    def test_similarity_fallback_to_unfiltered_when_filtered_empty(self) -> None:
        mgr = _make_manager({"USE_MMR": False})
        db = MagicMock()
        db.similarity_search_with_score.side_effect = [[], _make_scored_docs("fallback", score=0.5)]
        mgr._vector_dbs["col"] = db  # noqa: SLF001
        result = mgr._search_collection("col", "query", [{"uuid": "x"}, None])  # noqa: SLF001
        self.assertEqual(db.similarity_search_with_score.call_count, 2)
        self.assertEqual(result, ["fallback"])

    def test_use_mmr_default_is_true(self) -> None:
        """When USE_MMR is absent from config, MMR should be used (default True)."""
        mgr = _make_manager({})
        mgr.configs.pop("USE_MMR", None)
        db = MagicMock()
        db.max_marginal_relevance_search.return_value = _make_docs("mmr chunk")
        mgr._vector_dbs["col"] = db  # noqa: SLF001
        result = mgr._search_collection("col", "query", [None])  # noqa: SLF001
        db.max_marginal_relevance_search.assert_called_once()
        self.assertEqual(result, ["mmr chunk"])


if __name__ == "__main__":
    unittest.main()
