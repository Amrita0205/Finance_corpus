"""
finrag/retrieval/base_retriever.py
===================================
Abstract base class for all retrieval strategies.
Concrete implementations must override ``retrieve()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from finrag.storage.vector_store import FAISSVectorStore, SearchResult
from finrag.embeddings.embedding_model import EmbeddingModel


class BaseRetriever(ABC):
    """
    Abstract retriever.  All strategies share a vector store and an
    embedding model but differ in how the query is represented before search.
    """

    strategy_name: str = "base"

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_model: EmbeddingModel,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """
        Retrieve the most relevant documents for ``query``.

        Parameters
        ----------
        query:
            Raw user query string.
        top_k:
            Number of results to return.

        Returns
        -------
        List of SearchResult objects ordered by descending relevance score.
        """
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"store_docs={self.vector_store.total_documents})"
        )
