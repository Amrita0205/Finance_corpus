"""
finrag/pipeline/rag_pipeline.py
================================
RAGPipeline — main orchestration class.

Responsibilities
----------------
1. Ingest a list of document dicts (``id``, ``title``, ``content``).
2. Generate embeddings in batches.
3. Store vectors in the FAISSVectorStore.
4. Expose both retrieval strategies through a clean API.
5. Accept a ``GenerativeModel`` for Strategy B (injected for testability).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from finrag.embeddings.embedding_model import EmbeddingModel
from finrag.storage.vector_store import FAISSVectorStore, SearchResult
from finrag.retrieval.strategy_a import RawVectorRetriever
from finrag.retrieval.strategy_b import HyDERetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Context-Aware Retrieval Engine for the Finance Domain.

    Parameters
    ----------
    embedding_model:
        EmbeddingModel instance (wraps sentence-transformers / Vertex AI gecko).
    generative_model:
        Any object with ``generate_content(prompt) → response`` where
        ``response.text`` is a string.  Pass MockGenerativeModel for offline
        use or the real ``vertexai.generative_models.GenerativeModel`` in prod.
    use_hybrid_hyde:
        When True, Strategy B averages the raw query and HyDE vectors.
    batch_size:
        Number of documents encoded per embedding call (controls GPU/CPU memory).

    Example
    -------
    >>> from data.finance_corpus import FINANCE_CORPUS
    >>> from finrag.mocks.vertexai_mocks import MockGenerativeModel
    >>> pipeline = RAGPipeline(generative_model=MockGenerativeModel())
    >>> pipeline.ingest(FINANCE_CORPUS)
    >>> results = pipeline.retrieve_strategy_a("peak load handling", top_k=3)
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        generative_model: Any = None,
        use_hybrid_hyde: bool = False,
        batch_size: int = 32,
    ) -> None:
        self.embedding_model = embedding_model or EmbeddingModel()
        self.use_hybrid_hyde = use_hybrid_hyde
        self.batch_size = batch_size
        self._vector_store = FAISSVectorStore()

        # Generative model: fall back to mock if none provided
        if generative_model is None:
            from finrag.mocks.vertexai_mocks import MockGenerativeModel
            generative_model = MockGenerativeModel()
        self._generative_model = generative_model

        # Build retriever instances (lazy; populated after ingestion)
        self._retriever_a: RawVectorRetriever | None = None
        self._retriever_b: HyDERetriever | None = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, documents: list[dict]) -> None:
        """
        Embed and store a list of documents.

        Parameters
        ----------
        documents:
            List of dicts with keys ``id``, ``title``, ``content``.
            Any extra keys are preserved in metadata but not embedded.
        """
        if not documents:
            raise ValueError("documents list must not be empty.")

        required_keys = {"id", "title", "content"}
        for i, doc in enumerate(documents):
            missing = required_keys - doc.keys()
            if missing:
                raise ValueError(
                    f"Document at index {i} is missing required keys: {missing}"
                )

        logger.info("Ingesting %d documents in batches of %d …", len(documents), self.batch_size)

        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            texts = [doc["content"] for doc in batch]
            batch_emb = self.embedding_model.embed_documents(texts)
            all_embeddings.append(batch_emb)
            logger.debug("  Embedded batch [%d:%d]", start, start + len(batch))

        embeddings_matrix = np.vstack(all_embeddings)

        metadata = [
            {"id": doc["id"], "title": doc["title"], "content": doc["content"]}
            for doc in documents
        ]
        self._vector_store.add_documents(embeddings_matrix, metadata)

        # (Re)build retriever objects pointing to the now-populated store
        self._retriever_a = RawVectorRetriever(
            vector_store=self._vector_store,
            embedding_model=self.embedding_model,
        )
        self._retriever_b = HyDERetriever(
            vector_store=self._vector_store,
            embedding_model=self.embedding_model,
            generative_model=self._generative_model,
            use_hybrid=self.use_hybrid_hyde,
        )

        logger.info(
            "Ingestion complete. %s", self._vector_store
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _assert_ingested(self) -> None:
        if self._retriever_a is None:
            raise RuntimeError("Pipeline has not been ingested. Call ingest() first.")

    def retrieve_strategy_a(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Strategy A: raw vector search."""
        self._assert_ingested()
        return self._retriever_a.retrieve(query, top_k=top_k)

    def retrieve_strategy_b(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Strategy B: HyDE AI-enhanced retrieval."""
        self._assert_ingested()
        return self._retriever_b.retrieve(query, top_k=top_k)

    def retrieve_both(
        self, query: str, top_k: int = 3
    ) -> dict[str, list[SearchResult]]:
        """
        Run both strategies and return a dict with keys
        ``"strategy_a"`` and ``"strategy_b"``.
        """
        return {
            "strategy_a": self.retrieve_strategy_a(query, top_k=top_k),
            "strategy_b": self.retrieve_strategy_b(query, top_k=top_k),
        }

    def get_hyde_expansion(self, query: str) -> str:
        """Return the hypothetical document generated for a query (Strategy B)."""
        self._assert_ingested()
        return self._retriever_b.get_expansion(query)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_path: str, meta_path: str) -> None:
        """Persist the vector store to disk."""
        self._vector_store.save(index_path, meta_path)

    @classmethod
    def load(
        cls,
        index_path: str,
        meta_path: str,
        embedding_model: EmbeddingModel | None = None,
        generative_model: Any = None,
    ) -> "RAGPipeline":
        """Re-hydrate a pipeline from a previously saved vector store."""
        pipeline = cls(
            embedding_model=embedding_model,
            generative_model=generative_model,
        )
        pipeline._vector_store = FAISSVectorStore.load(index_path, meta_path)
        pipeline._retriever_a = RawVectorRetriever(
            pipeline._vector_store, pipeline.embedding_model
        )
        pipeline._retriever_b = HyDERetriever(
            pipeline._vector_store,
            pipeline.embedding_model,
            pipeline._generative_model,
        )
        return pipeline

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vector_store(self) -> FAISSVectorStore:
        return self._vector_store

    def __repr__(self) -> str:
        return (
            f"RAGPipeline("
            f"docs={self._vector_store.total_documents}, "
            f"hybrid_hyde={self.use_hybrid_hyde})"
        )
