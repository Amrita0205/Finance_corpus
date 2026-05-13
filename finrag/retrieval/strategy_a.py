"""
finrag/retrieval/strategy_a.py
================================
Strategy A — Raw Vector Search
--------------------------------
The query is embedded directly using the embedding model and then searched
against the FAISS index.  No query transformation is applied.

This is the baseline retrieval strategy and corresponds to a standard
dense retrieval pipeline (e.g., DPR, bi-encoder).
"""

from __future__ import annotations

import logging

from finrag.retrieval.base_retriever import BaseRetriever
from finrag.storage.vector_store import SearchResult

logger = logging.getLogger(__name__)


class RawVectorRetriever(BaseRetriever):
    """
    Strategy A: embed the raw user query → cosine search in FAISS.

    Strengths
    ---------
    * Fast, deterministic.
    * Zero additional API calls.

    Limitations
    -----------
    * Short or ambiguous queries may have embeddings far from the relevant
      document embeddings ("vocabulary mismatch").
    * Cannot leverage domain context not present in the original query tokens.
    """

    strategy_name = "Strategy A — Raw Vector Search"

    def retrieve(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """
        Embed ``query`` directly and search the vector store.

        Parameters
        ----------
        query:
            Raw user query string.
        top_k:
            Number of top results to return.

        Returns
        -------
        List[SearchResult] ordered by descending cosine similarity.
        """
        logger.debug("[Strategy A] Embedding raw query: %r", query)
        query_vec = self.embedding_model.embed_query(query)
        results = self.vector_store.search(query_vec, top_k=top_k)
        logger.info("[Strategy A] Retrieved %d results for query: %r", len(results), query)
        return results
