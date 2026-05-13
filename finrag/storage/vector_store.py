"""
finrag/storage/vector_store.py
================================
FAISS-backed vector store with cosine similarity.

Similarity Metric Choice: Cosine vs Euclidean
----------------------------------------------
We use **cosine similarity** (via dot product on L2-normalised vectors) for the
following reasons:

1. **Magnitude invariance**: Financial text chunks vary in length (sentence vs
   paragraph). Cosine similarity is unaffected by vector magnitude, so a
   short 1-sentence summary competes fairly with a long multi-clause paragraph.

2. **Semantic similarity tasks**: Sentence-transformers are trained with cosine
   loss; raw Euclidean distance on their embeddings underperforms (confirmed in
   the SBERT paper, Reimers & Gurevych 2019).

3. **FAISS efficiency**: ``IndexFlatIP`` (inner product on L2-normalised vectors)
   gives exact cosine search with O(n·d) complexity — perfectly adequate for
   corpora of up to ~1M chunks without approximation errors.

4. **Euclidean (L2) drawback**: IndexFlatL2 gives different rankings for the same
   semantic content if embedding norms differ; also, its geometric distance does
   not correlate linearly with human-perceived semantic distance.

Vertex AI Vector Search (Matching Engine) Migration Path
---------------------------------------------------------
When moving to production on GCP:

1. **Index creation**: POST to ``projects/{project}/locations/{region}/indexes``
   with ``config.algorithmConfig.treeAhConfig`` (ScaNN) or
   ``config.algorithmConfig.bruteForceConfig`` for small corpora.
   Set ``distanceMeasureType: "COSINE_DISTANCE"`` (maps to our cosine choice).

2. **Batch upsert**: Call ``IndexService.UpsertDatapoints`` with your normalised
   float32 vectors and string IDs.

3. **Online query**: Deploy the index to an ``IndexEndpoint`` and call
   ``MatchService.FindNeighbors`` with the query embedding and ``neighbor_count``.

4. **Filtering**: Matching Engine supports numeric/string restricts that can
   filter by metadata fields (e.g., document date, asset class) — equivalent
   to the ``filter_ids`` parameter here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single retrieval result returned by the vector store."""

    doc_id: str
    title: str
    content: str
    score: float  # cosine similarity ∈ [−1, 1]; higher = more similar
    rank: int


class FAISSVectorStore:
    """
    In-memory FAISS vector store using cosine similarity.

    The store holds:
    * A FAISS IndexFlatIP index (exact inner-product = cosine on unit vectors)
    * A metadata list aligned by integer index with FAISS positions

    Parameters
    ----------
    embedding_dim:
        Dimensionality of vectors to be stored.  If not given, inferred on
        first ``add_documents`` call.
    """

    def __init__(self, embedding_dim: int | None = None) -> None:
        self._dim = embedding_dim
        self._index: Any = None  # faiss.IndexFlatIP
        self._metadata: list[dict] = []  # aligned with FAISS positions

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _build_index(self, dim: int) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            ) from exc
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        logger.info("Initialised FAISS IndexFlatIP with dim=%d", dim)

    def add_documents(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
    ) -> None:
        """
        Add a batch of pre-computed (and L2-normalised) embeddings to the index.

        Parameters
        ----------
        embeddings:
            Float32 array of shape (n_docs, embedding_dim). Must be L2-normalised.
        metadata:
            List of dicts (length == n_docs) with keys: id, title, content.
        """
        assert len(embeddings) == len(metadata), (
            f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) must align."
        )
        if self._index is None:
            self._build_index(embeddings.shape[1])

        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add(vectors)
        self._metadata.extend(metadata)
        logger.info("Added %d documents. Total: %d", len(metadata), len(self._metadata))

    def clear(self) -> None:
        """Remove all vectors and metadata from the store."""
        if self._index is not None:
            self._index.reset()
        self._metadata.clear()
        logger.info("Vector store cleared.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        filter_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve the top-k most similar documents.

        Parameters
        ----------
        query_vector:
            1-D float32 array of shape (embedding_dim,). Should be L2-normalised.
        top_k:
            Number of results to return.
        filter_ids:
            Optional allow-list of document IDs. Results are filtered to these
            IDs post-retrieval (brute-force; suitable for small corpora).

        Returns
        -------
        List of SearchResult sorted descending by cosine score.
        """
        if self._index is None or self._index.ntotal == 0:
            raise ValueError("Vector store is empty. Call add_documents() first.")

        qvec = np.ascontiguousarray(
            query_vector.reshape(1, -1), dtype=np.float32
        )
        # Retrieve extra results if we need to filter
        k = min(top_k * 5 if filter_ids else top_k, self._index.ntotal)
        scores, indices = self._index.search(qvec, k)

        results: list[SearchResult] = []
        rank = 1
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx]
            if filter_ids and meta["id"] not in filter_ids:
                continue
            results.append(
                SearchResult(
                    doc_id=meta["id"],
                    title=meta["title"],
                    content=meta["content"],
                    score=float(score),
                    rank=rank,
                )
            )
            rank += 1
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, index_path: str, meta_path: str) -> None:
        """Save FAISS index and metadata to disk."""
        import faiss  # type: ignore
        import json

        faiss.write_index(self._index, index_path)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(self._metadata, fh, ensure_ascii=False, indent=2)
        logger.info("Saved index to %s and metadata to %s", index_path, meta_path)

    @classmethod
    def load(cls, index_path: str, meta_path: str) -> "FAISSVectorStore":
        """Load a previously saved vector store from disk."""
        import faiss  # type: ignore
        import json

        store = cls()
        store._index = faiss.read_index(index_path)
        store._dim = store._index.d
        with open(meta_path, encoding="utf-8") as fh:
            store._metadata = json.load(fh)
        logger.info(
            "Loaded index from %s (%d docs)", index_path, store._index.ntotal
        )
        return store

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def total_documents(self) -> int:
        return len(self._metadata)

    def __repr__(self) -> str:
        return (
            f"FAISSVectorStore(dim={self._dim}, "
            f"total_documents={self.total_documents})"
        )
