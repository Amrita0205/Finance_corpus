"""
finrag/embeddings/embedding_model.py
=====================================
Thin wrapper that presents a stable interface regardless of whether the
real Vertex AI SDK is available.  In production, swap ``MockTextEmbeddingModel``
for the real ``vertexai.language_models.TextEmbeddingModel``.

Design Notes
------------
* Vectors are **L2-normalised** before being returned so that inner-product
  similarity equals cosine similarity — this lets us use FAISS IndexFlatIP
  (exact inner product) as a fast cosine search index.
* The encode() / embed_documents() / embed_query() interface mirrors both
  LangChain's Embeddings protocol and the raw sentence-transformers API so
  the module is easily swappable in larger pipelines.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Unified embedding interface wrapping ``MockTextEmbeddingModel``.

    Parameters
    ----------
    model_name:
        The Vertex AI model identifier (used for documentation / mock routing).
    normalise:
        If True (default), L2-normalise every embedding vector so that
        cosine similarity ≡ dot product.  Required for IndexFlatIP in FAISS.
    """

    DEFAULT_VERTEX_MODEL = "textembedding-gecko@003"
    LOCAL_ST_MODEL = "local-tfidf-svd-384 (sklearn TF-IDF + TruncatedSVD)"

    def __init__(
        self,
        model_name: str = DEFAULT_VERTEX_MODEL,
        normalise: bool = True,
    ) -> None:
        self.model_name = model_name
        self.normalise = normalise
        self._backend: Any = None  # lazy-load

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_backend(self) -> None:
        """Lazily initialise the embedding backend."""
        if self._backend is not None:
            return
        try:
            # Try real Vertex AI SDK first (will succeed in production)
            from vertexai.language_models import TextEmbeddingModel  # type: ignore
            self._backend = TextEmbeddingModel.from_pretrained(self.model_name)
            logger.info("Using real Vertex AI TextEmbeddingModel (%s)", self.model_name)
        except (ImportError, Exception):
            # Fall back to the offline mock backed by sentence-transformers
            from finrag.mocks.vertexai_mocks import MockTextEmbeddingModel
            self._backend = MockTextEmbeddingModel.from_pretrained(self.model_name)
            logger.info(
                "Vertex AI SDK unavailable — using MockTextEmbeddingModel "
                "backed by %s",
                self.LOCAL_ST_MODEL,
            )

    def _l2_normalise(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid zero-division
        return matrix / norms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Return embedding dimension by encoding a dummy string."""
        return self.embed_query("dimension probe").shape[0]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of documents.

        Parameters
        ----------
        texts:
            List of plain-text strings to embed.

        Returns
        -------
        np.ndarray of shape (len(texts), embedding_dim), dtype=float32.
        Vectors are L2-normalised when ``self.normalise=True``.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        self._load_backend()
        embeddings = self._backend.get_embeddings(texts)
        matrix = np.array([e.values for e in embeddings], dtype=np.float32)
        if self.normalise:
            matrix = self._l2_normalise(matrix)
        logger.debug("Embedded %d documents → shape %s", len(texts), matrix.shape)
        return matrix

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns
        -------
        np.ndarray of shape (embedding_dim,), dtype=float32.
        """
        matrix = self.embed_documents([query])
        return matrix[0]

    # Alias for LangChain compatibility
    encode = embed_documents
