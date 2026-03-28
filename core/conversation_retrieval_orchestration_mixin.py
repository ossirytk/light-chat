import re

from loguru import logger

from core.retrieval_keys import build_where_filters
from core.retrieval_shared_types import RetrievalTrace, WhereFilter


class ConversationRetrievalOrchestrationMixin:
    def _log_retrieval_telemetry(
        self,
        query: str,
        main_trace: RetrievalTrace,
        mes_trace: RetrievalTrace,
        cleanup_stats: dict[str, int],
    ) -> None:
        if not self.runtime_config.rag_telemetry_enabled:
            return

        logger.info(
            "RAG telemetry: query_chars={} main(mode={} filter={} cand={} out={} spread={}) "
            "mes(mode={} filter={} cand={} out={} spread={}) cleanup(main={} mes={} cross_removed={})",
            len(query),
            main_trace.get("mode", "unknown"),
            main_trace.get("filter_path", "n/a"),
            main_trace.get("candidates", 0),
            main_trace.get("returned", 0),
            main_trace.get("score_spread", "n/a"),
            mes_trace.get("mode", "unknown"),
            mes_trace.get("filter_path", "n/a"),
            mes_trace.get("candidates", 0),
            mes_trace.get("returned", 0),
            mes_trace.get("score_spread", "n/a"),
            cleanup_stats.get("main", 0),
            cleanup_stats.get("mes", 0),
            cleanup_stats.get("cross_removed", 0),
        )

    def _search_collection_with_trace(
        self,
        collection_name: str,
        query: str,
        filters: list[WhereFilter],
        k: int | None = None,
    ) -> tuple[list[str], RetrievalTrace]:
        trace: RetrievalTrace = {
            "mode": "mmr" if self.runtime_config.use_mmr else "similarity",
            "filter_path": "none",
            "candidates": 0,
            "returned": 0,
            "score_spread": None,
            "rerank_applied": False,
        }

        if not query:
            return [], trace

        if k is None:
            k = self.rag_k

        retrieval_k = max(k, self.runtime_config.rag_rerank_top_n) if self.runtime_config.rag_rerank_enabled else k
        db = self._get_vector_db(collection_name)
        use_mmr = self.runtime_config.use_mmr
        fetch_k = self.runtime_config.rag_fetch_k
        lambda_mult = self.runtime_config.lambda_mult
        score_threshold = self.configs.get("RAG_SCORE_THRESHOLD")
        candidate_chunks: list[str] = []
        similarity_scores: list[float] = []

        for where in filters:
            trace["filter_path"] = self._describe_where_filter(where)
            if use_mmr:
                candidate_chunks = self._run_mmr_search(
                    db,
                    query,
                    where,
                    {
                        "retrieval_k": retrieval_k,
                        "fetch_k": max(retrieval_k, fetch_k),
                        "lambda_mult": lambda_mult,
                    },
                )
                if candidate_chunks or where is None:
                    break
                continue

            candidate_chunks, similarity_scores = self._run_similarity_search(
                db,
                query,
                where,
                retrieval_k,
                score_threshold,
            )
            if candidate_chunks or where is None:
                break

        result_chunks = candidate_chunks[:k]
        if self.runtime_config.rag_rerank_enabled and candidate_chunks:
            result_chunks = self._rerank_chunks(query=query, chunks=candidate_chunks, k=k)
            trace["rerank_applied"] = True

        score_min, score_max, score_spread = self._score_spread(similarity_scores)
        trace["candidates"] = len(candidate_chunks)
        trace["returned"] = len(result_chunks)
        trace["score_min"] = score_min
        trace["score_max"] = score_max
        trace["score_spread"] = score_spread

        return result_chunks, trace

    def _should_skip_rag_for_message(self, message: str) -> bool:
        """Return True for short small-talk turns where lore retrieval hurts quality."""
        normalized = re.sub(r"\s+", " ", message).strip()
        if not normalized:
            return True
        words = [word for word in normalized.split(" ") if word]
        max_words = self.runtime_config.small_talk_max_words
        if len(words) > max_words:
            return False
        return any(pattern.search(normalized) for pattern in self._SMALL_TALK_PATTERNS)

    def _should_skip_rag_for_followup(self, message: str) -> bool:
        """Skip RAG on short follow-up chat turns without lore/entity matches."""
        normalized = re.sub(r"\s+", " ", message).strip()
        if not normalized:
            return True
        max_words = self.runtime_config.followup_rag_max_words
        words = [word for word in normalized.split(" ") if word]
        if len(words) > max_words:
            return False
        if not self.rag_collection:
            return True
        matches = self._get_key_matches(normalized, self.rag_collection)
        return len(matches) == 0

    def _search_collection(
        self,
        collection_name: str,
        query: str,
        filters: list[WhereFilter],
        k: int | None = None,
    ) -> list[str]:
        chunks, _trace = self._search_collection_with_trace(collection_name, query, filters, k=k)
        return chunks

    def _query_terms(self, query: str) -> set[str]:
        terms = re.findall(r"[a-zA-Z0-9_]+", query.lower())
        return {term for term in terms if len(term) > self._MIN_QUERY_TERM_LEN and term not in self._QUERY_STOPWORDS}

    def _build_multi_queries(self, query: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", query).strip()
        if not normalized:
            return []

        max_variants = max(1, self.runtime_config.rag_multi_query_max_variants)
        variants: list[str] = [normalized]
        terms = sorted(self._query_terms(normalized))

        # Lexical reformulations that preserve intent while giving retrieval multiple angles.
        compact_query_term_count = self._COMPACT_QUERY_TERM_COUNT
        if len(terms) >= compact_query_term_count:
            variants.append(" ".join(terms[:compact_query_term_count]))
            variants.append(" ".join(terms[-compact_query_term_count:]))
        elif terms:
            variants.append(" ".join(terms))

        # Keep deterministic order and uniqueness.
        deduped: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            cleaned = variant.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
            if len(deduped) >= max_variants:
                break
        return deduped

    def _merge_multi_query_chunks(self, ranked_results: list[list[str]], k: int) -> list[str]:
        if not ranked_results:
            return []

        merged: list[str] = []
        seen_signatures: set[str] = set()
        max_depth = max(len(result) for result in ranked_results)

        for index in range(max_depth):
            for result in ranked_results:
                if index >= len(result):
                    continue
                chunk = result[index].strip()
                if not chunk:
                    continue
                signature = self._chunk_signature(chunk)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                merged.append(chunk)
                if len(merged) >= k:
                    return merged
        return merged

    def _multi_query_search_with_trace(
        self,
        collection_name: str,
        base_query: str,
        filters: list[WhereFilter],
        k: int,
    ) -> tuple[list[str], RetrievalTrace]:
        queries = [base_query]
        if self.runtime_config.rag_multi_query_enabled:
            queries = self._build_multi_queries(base_query)

        if not queries:
            return [], {
                "mode": "mmr" if self.runtime_config.use_mmr else "similarity",
                "filter_path": "none",
                "candidates": 0,
                "returned": 0,
                "score_spread": None,
                "rerank_applied": False,
                "queries": 0,
            }

        traces: list[RetrievalTrace] = []
        ranked_results: list[list[str]] = []
        for retrieval_query in queries:
            chunks, trace = self._search_collection_with_trace(collection_name, retrieval_query, filters, k=k)
            traces.append(trace)
            ranked_results.append(chunks)

        merged_chunks = self._merge_multi_query_chunks(ranked_results, k)
        aggregate_trace = traces[0] if traces else {}
        aggregate_trace = {
            **aggregate_trace,
            "queries": len(queries),
            "candidates": sum(int(trace.get("candidates", 0)) for trace in traces),
            "returned": len(merged_chunks),
            "multi_query_enabled": self.runtime_config.rag_multi_query_enabled,
        }
        return merged_chunks, aggregate_trace

    def _get_vector_context(self, query: str, k: int | None = None, *, include_mes: bool = True) -> tuple[str, str]:
        if not self.rag_collection:
            return "", ""
        # Enrich query with character name to orient embedding search toward the character domain
        enriched_query = f"{self.character_name} {query}" if self.character_name else query
        matches = self._get_key_matches(query, self.rag_collection)
        filters = build_where_filters(matches)
        retrieval_k = k if k is not None else self.rag_k
        context_chunks, context_trace = self._multi_query_search_with_trace(
            self.rag_collection, enriched_query, filters, k=retrieval_k
        )
        # Use unfiltered search for message examples: goal is stylistic match, not factual (Section 6.2)
        if include_mes:
            k_mes = k if k is not None else self.rag_k_mes
            mes_chunks, mes_trace = self._multi_query_search_with_trace(
                f"{self.rag_collection}_mes",
                enriched_query,
                [None],
                k=k_mes,
            )
        else:
            mes_chunks, mes_trace = (
                [],
                {
                    "mode": "disabled",
                    "filter_path": "none",
                    "candidates": 0,
                    "returned": 0,
                    "queries": 0,
                    "rerank_applied": False,
                },
            )
        context_chunks = self._filter_context_chunks(context_chunks)
        mes_chunks = self._filter_context_chunks(mes_chunks)
        context_chunks, mes_chunks, cross_removed = self._dedupe_cross_collection_chunks(context_chunks, mes_chunks)
        self.last_retrieval_debug = {
            "collection": self.rag_collection,
            "key_match_count": len(matches),
            "main": {
                "mode": str(context_trace.get("mode", "unknown")),
                "filter_path": str(context_trace.get("filter_path", "none")),
                "candidates": int(context_trace.get("candidates", 0) or 0),
                "returned": int(context_trace.get("returned", 0) or 0),
                "queries": int(context_trace.get("queries", 0) or 0),
                "rerank_applied": bool(context_trace.get("rerank_applied", False)),
            },
            "mes": {
                "mode": str(mes_trace.get("mode", "unknown")),
                "filter_path": str(mes_trace.get("filter_path", "none")),
                "candidates": int(mes_trace.get("candidates", 0) or 0),
                "returned": int(mes_trace.get("returned", 0) or 0),
                "queries": int(mes_trace.get("queries", 0) or 0),
                "rerank_applied": bool(mes_trace.get("rerank_applied", False)),
            },
            "cleanup": {
                "main": len(context_chunks),
                "mes": len(mes_chunks),
                "cross_removed": cross_removed,
            },
        }
        self._log_retrieval_telemetry(
            query=enriched_query,
            main_trace=context_trace,
            mes_trace=mes_trace,
            cleanup_stats={"main": len(context_chunks), "mes": len(mes_chunks), "cross_removed": cross_removed},
        )
        vector_context = "\n\n".join(context_chunks)
        vector_context = self._dedupe_chunk_sections(vector_context)
        vector_context = self._cap_context_text(vector_context)
        vector_context = self._compress_context_sentences(query=enriched_query, context=vector_context)
        vector_context = self._cap_context_text(vector_context)
        mes_example = "\n\n".join(mes_chunks)
        return vector_context, mes_example
