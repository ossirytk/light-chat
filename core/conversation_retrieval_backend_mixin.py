from loguru import logger

from core.rag_dependencies import import_cross_encoder, import_vector_dependencies
from core.retrieval_shared_types import WhereFilter

EMBEDDING_MODEL_METADATA_KEY = "embedding:model"
EMBEDDING_DIMENSION_METADATA_KEY = "embedding:dimension"
EMBEDDING_NORMALIZE_METADATA_KEY = "embedding:normalize"


class ConversationRetrievalBackendMixin:
    def _get_vector_client(self) -> object:
        if self._vector_client is None:
            chromadb_module, settings_cls, _chroma_cls, _huggingface_embeddings_cls = import_vector_dependencies()
            self._vector_client = chromadb_module.PersistentClient(
                path=self.persist_directory,
                settings=settings_cls(anonymized_telemetry=False),
            )
        return self._vector_client

    def _get_vector_embedder(self) -> object:
        if self._vector_embedder is None:
            _chromadb_module, _settings_cls, _chroma_cls, huggingface_embeddings_cls = import_vector_dependencies()
            model_kwargs = {"device": self.runtime_config.embedding_device}
            encode_kwargs = {"normalize_embeddings": True}
            self._vector_embedder = huggingface_embeddings_cls(
                model_name=self.runtime_config.embedding_model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                cache_folder=self.embedding_cache,
            )
        return self._vector_embedder

    def _get_vector_db(self, collection_name: str) -> object:
        if collection_name not in self._vector_dbs:
            _chromadb_module, _settings_cls, chroma_cls, _huggingface_embeddings_cls = import_vector_dependencies()
            self._assert_collection_embedding_fingerprint(collection_name)
            self._vector_dbs[collection_name] = chroma_cls(
                client=self._get_vector_client(),
                collection_name=collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self._get_vector_embedder(),
            )
        return self._vector_dbs[collection_name]

    def _infer_embedding_dimension(self) -> int | None:
        cached_dimension = getattr(self, "_embedding_dimension", None)
        if isinstance(cached_dimension, int) and cached_dimension > 0:
            return cached_dimension

        try:
            vector = self._get_vector_embedder().embed_query("dimension_probe")
        except Exception as error:
            logger.warning("Could not infer embedding dimension for runtime fingerprint checks: {}", error)
            return None

        if not isinstance(vector, list):
            return None

        dimension = len(vector)
        self._embedding_dimension = dimension
        return dimension

    def _expected_embedding_fingerprint(self) -> dict[str, object]:
        fingerprint: dict[str, object] = {
            EMBEDDING_MODEL_METADATA_KEY: self.runtime_config.embedding_model,
            EMBEDDING_NORMALIZE_METADATA_KEY: True,
        }
        embedding_dimension = self._infer_embedding_dimension()
        if embedding_dimension is not None:
            fingerprint[EMBEDDING_DIMENSION_METADATA_KEY] = embedding_dimension
        return fingerprint

    def _assert_collection_embedding_fingerprint(self, collection_name: str) -> None:
        try:
            collection = self._get_vector_client().get_collection(collection_name)
        except ValueError:
            return

        metadata = collection.metadata or {}
        expected_fingerprint = self._expected_embedding_fingerprint()
        mismatches = [
            (key, metadata[key], expected_fingerprint[key])
            for key in expected_fingerprint
            if key in metadata and metadata[key] != expected_fingerprint[key]
        ]
        if not mismatches:
            return

        mismatch_summary = ", ".join(
            f"{key}: existing={actual!r} expected={expected!r}" for key, actual, expected in mismatches
        )
        msg = (
            f"Collection '{collection_name}' has incompatible embedding fingerprint; refusing mixed-model read. "
            f"{mismatch_summary}"
        )
        raise RuntimeError(msg)

    def _get_cross_encoder(self) -> object:
        if self._cross_encoder is None:
            cross_encoder_cls = import_cross_encoder()
            self._cross_encoder = cross_encoder_cls(self.runtime_config.rag_rerank_model, device="cpu")
        return self._cross_encoder

    def _rerank_chunks(self, query: str, chunks: list[str], k: int) -> list[str]:
        if not chunks:
            return []

        top_n = max(k, self.runtime_config.rag_rerank_top_n)
        candidates = chunks[:top_n]
        if len(candidates) <= 1:
            return chunks[:k]

        try:
            cross_encoder = self._get_cross_encoder()
            pairs = [(query, chunk) for chunk in candidates]
            scores = cross_encoder.predict(pairs, show_progress_bar=False)
            scored_chunks = list(zip(candidates, scores, strict=False))
            scored_chunks.sort(key=lambda item: float(item[1]), reverse=True)
            reranked = [chunk for chunk, _score in scored_chunks]
            if len(chunks) > len(candidates):
                reranked.extend(chunks[len(candidates) :])
            return reranked[:k]
        except Exception as error:
            logger.warning("Reranking failed; using first-pass retrieval order. Error: {}", error)
            return chunks[:k]

    @staticmethod
    def _score_spread(scores: list[float]) -> tuple[float | None, float | None, float | None]:
        if not scores:
            return None, None, None
        min_score = min(scores)
        max_score = max(scores)
        return min_score, max_score, max_score - min_score

    @staticmethod
    def _run_mmr_search(
        db: object,
        query: str,
        where: WhereFilter,
        search_options: dict[str, object],
    ) -> list[str]:
        search_kwargs: dict[str, object] = {
            "query": query,
            "k": int(search_options["retrieval_k"]),
            "fetch_k": int(search_options["fetch_k"]),
            "lambda_mult": float(search_options["lambda_mult"]),
        }
        if where is not None:
            search_kwargs["filter"] = where
        docs = db.max_marginal_relevance_search(**search_kwargs)
        return [doc.page_content for doc in docs]

    @staticmethod
    def _run_similarity_search(
        db: object,
        query: str,
        where: WhereFilter,
        retrieval_k: int,
        score_threshold: object,
    ) -> tuple[list[str], list[float]]:
        search_kwargs: dict[str, object] = {"query": query, "k": retrieval_k}
        if where is not None:
            search_kwargs["filter"] = where

        docs = db.similarity_search_with_score(**search_kwargs)
        if score_threshold is not None:
            threshold_value = float(score_threshold)
            docs = [(doc, score) for doc, score in docs if score <= threshold_value]

        chunks = [doc.page_content for doc, _score in docs]
        scores = [float(score) for _doc, score in docs]
        return chunks, scores
