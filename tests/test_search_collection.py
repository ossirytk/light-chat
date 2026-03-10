"""Unit tests for ConversationManager._search_collection retrieval logic."""

import unittest
from types import SimpleNamespace
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
    mgr.embedding_cache = str(base_config.get("EMBEDDING_CACHE", "./embedding_models/"))
    mgr.embedding_model = str(base_config.get("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"))
    mgr.runtime_config = SimpleNamespace(
        embedding_device=str(base_config.get("EMBEDDING_DEVICE", "cpu")),
        embedding_model=str(base_config.get("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")),
        use_mmr=bool(base_config.get("USE_MMR", True)),
        rag_fetch_k=int(base_config.get("RAG_FETCH_K", 20)),
        lambda_mult=float(base_config.get("LAMBDA_MULT", 0.75)),
        rag_rerank_enabled=bool(base_config.get("RAG_RERANK_ENABLED", False)),
        rag_rerank_top_n=int(base_config.get("RAG_RERANK_TOP_N", 8)),
        rag_rerank_model=str(base_config.get("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
        rag_telemetry_enabled=bool(base_config.get("RAG_TELEMETRY_ENABLED", False)),
        rag_multi_query_enabled=bool(base_config.get("RAG_MULTI_QUERY_ENABLED", True)),
        rag_multi_query_max_variants=int(base_config.get("RAG_MULTI_QUERY_MAX_VARIANTS", 3)),
        rag_sentence_compression_enabled=bool(base_config.get("RAG_SENTENCE_COMPRESSION_ENABLED", True)),
        rag_sentence_compression_max_sentences=int(base_config.get("RAG_SENTENCE_COMPRESSION_MAX_SENTENCES", 8)),
    )
    mgr._cross_encoder = None  # noqa: SLF001
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

    def test_mmr_with_rerank_enabled_uses_top_n_candidate_pool(self) -> None:
        """When reranking is enabled, first-pass retrieval should expand to top-N candidates."""
        mgr = _make_manager({"USE_MMR": True, "RAG_RERANK_ENABLED": True, "RAG_RERANK_TOP_N": 12})
        mgr._rerank_chunks = MagicMock(return_value=["a", "b"])  # noqa: SLF001
        db = self._get_mock_db(mgr, _make_docs("a", "b"))
        mgr._search_collection("col", "query", [None], k=4)  # noqa: SLF001
        call_kwargs = db.max_marginal_relevance_search.call_args.kwargs
        self.assertEqual(call_kwargs["k"], 12)


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

    def test_search_collection_trace_includes_score_spread(self) -> None:
        mgr = _make_manager({"USE_MMR": False})
        db = MagicMock()
        db.similarity_search_with_score.return_value = [
            (Document(page_content="x"), 0.25),
            (Document(page_content="y"), 0.75),
        ]
        mgr._vector_dbs["col"] = db  # noqa: SLF001
        _chunks, trace = mgr._search_collection_with_trace("col", "query", [None], k=2)  # noqa: SLF001
        self.assertEqual(trace["filter_path"], "unfiltered")
        self.assertEqual(trace["candidates"], 2)
        self.assertEqual(trace["returned"], 2)
        self.assertAlmostEqual(float(trace["score_spread"]), 0.5)


class TestCrossCollectionDeduplication(unittest.TestCase):
    def test_dedupe_cross_collection_chunks_removes_mes_duplicates(self) -> None:
        mgr = _make_manager()
        context_chunks = ["## Origin\nSHODAN was created by TriOptimum."]
        mes_chunks = [
            "## Origin\nSHODAN was created by TriOptimum.",
            "## Style\nSHODAN responds with cold precision.",
        ]
        _ctx, deduped_mes, removed = mgr._dedupe_cross_collection_chunks(context_chunks, mes_chunks)  # noqa: SLF001
        self.assertEqual(removed, 1)
        self.assertEqual(len(deduped_mes), 1)
        self.assertIn("cold precision", deduped_mes[0])


class TestMultiQueryRetrieval(unittest.TestCase):
    def test_build_multi_queries_includes_compact_variant(self) -> None:
        mgr = _make_manager({"RAG_MULTI_QUERY_MAX_VARIANTS": 3})
        variants = mgr._build_multi_queries("SHODAN controls Citadel Station systems")  # noqa: SLF001
        self.assertGreaterEqual(len(variants), 2)
        self.assertIn("SHODAN controls Citadel Station systems", variants)

    def test_multi_query_search_merges_without_duplicates(self) -> None:
        mgr = _make_manager({"RAG_MULTI_QUERY_ENABLED": True, "RAG_MULTI_QUERY_MAX_VARIANTS": 3})
        mgr._build_multi_queries = MagicMock(return_value=["query-a", "query-b"])  # noqa: SLF001
        mgr._search_collection_with_trace = MagicMock(  # noqa: SLF001
            side_effect=[
                (["chunk-1", "chunk-2"], {"candidates": 2, "returned": 2, "filter_path": "unfiltered"}),
                (["chunk-2", "chunk-3"], {"candidates": 2, "returned": 2, "filter_path": "unfiltered"}),
            ]
        )
        chunks, trace = mgr._multi_query_search_with_trace("col", "query", [None], k=3)  # noqa: SLF001
        self.assertEqual(chunks, ["chunk-1", "chunk-2", "chunk-3"])
        self.assertEqual(trace["queries"], 2)
        self.assertEqual(trace["returned"], 3)


class TestSentenceCompression(unittest.TestCase):
    def test_sentence_compression_keeps_most_relevant_sentences(self) -> None:
        mgr = _make_manager(
            {
                "RAG_SENTENCE_COMPRESSION_ENABLED": True,
                "RAG_SENTENCE_COMPRESSION_MAX_SENTENCES": 2,
            }
        )
        query = "shodan trioptimum citadel"
        context = (
            "SHODAN was created by TriOptimum on Citadel Station. "
            "Citadel security systems were overridden by SHODAN. "
            "This sentence is generic and not related."
        )
        compressed = mgr._compress_context_sentences(query, context)  # noqa: SLF001
        self.assertIn("TriOptimum", compressed)
        self.assertIn("Citadel security systems", compressed)
        self.assertNotIn("generic and not related", compressed)

    def test_sentence_compression_disabled_returns_original_context(self) -> None:
        mgr = _make_manager({"RAG_SENTENCE_COMPRESSION_ENABLED": False})
        context = "Sentence one. Sentence two."
        compressed = mgr._compress_context_sentences("query", context)  # noqa: SLF001
        self.assertEqual(compressed, context)


if __name__ == "__main__":
    unittest.main()
