"""
finrag/retrieval/strategy_b.py
================================
Strategy B — AI-Enhanced Retrieval via HyDE Query Expansion
-------------------------------------------------------------
Implements **HyDE (Hypothetical Document Embeddings)**:

  1. The raw query is sent to a generative model (mocked Vertex AI Gemini /
     PaLM 2) which generates a *hypothetical* document paragraph — a fluent,
     domain-rich passage that a relevant document might contain.
  2. The hypothetical document is embedded (not the original query).
  3. The resulting vector is searched against the FAISS index.

Why HyDE outperforms raw-query embedding in domain-specific corpora
-------------------------------------------------------------------
* Embedding models map terse queries and long passages to different regions of
  the vector space. A query like "How does the system handle peak load?" sits far
  from the embedding of the full HFT paragraph it matches.
* The hypothetical document is already in *document space* — it uses the same
  vocabulary, sentence structure, and domain terminology as real corpus chunks,
  so its embedding is geometrically closer to the ground-truth match.
* This technique is described in: Gao et al. (2022) "Precise Zero-Shot Dense
  Retrieval without Relevance Labels" (https://arxiv.org/abs/2212.10496).

Hybrid variant (configurable)
------------------------------
When ``use_hybrid=True``, the final query vector is the average of the raw
query embedding and the hypothetical-document embedding.  This preserves the
original query intent while also pulling toward document space — useful when
the generative model's expansion diverges too far from the query.

Reference
---------
Gao, L., Ma, X., Lin, J., & Callan, J. (2022). *Precise Zero-Shot Dense
Retrieval without Relevance Labels*. arXiv:2212.10496.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from finrag.retrieval.base_retriever import BaseRetriever
from finrag.storage.vector_store import SearchResult

logger = logging.getLogger(__name__)


class HyDERetriever(BaseRetriever):
    """
    Strategy B: generate a hypothetical document → embed it → cosine search.

    Parameters
    ----------
    vector_store, embedding_model:
        Inherited from BaseRetriever.
    generative_model:
        An instance of MockGenerativeModel (or the real Vertex AI
        GenerativeModel). Must expose ``generate_content(prompt) → response``
        where ``response.text`` is the generated string.
    use_hybrid:
        If True, average the raw query embedding and the HyDE embedding before
        searching.  Default False (pure HyDE).
    hyde_prompt_template:
        A format string with a single ``{query}`` placeholder, used to
        instruct the generative model.
    """

    strategy_name = "Strategy B — AI-Enhanced HyDE Retrieval"

    DEFAULT_PROMPT = (
        "You are a senior quantitative finance engineer. "
        "Write a short technical paragraph (≤100 words) describing the answer to:\n\n"
        "{query}\n\n"
        "Write as if you are a document in a financial systems knowledge base. "
        "Use precise technical and financial terminology. Do not say 'I' or 'the answer is'."
    )

    def __init__(
        self,
        vector_store,
        embedding_model,
        generative_model: Any,
        use_hybrid: bool = False,
        hyde_prompt_template: str | None = None,
    ) -> None:
        super().__init__(vector_store, embedding_model)
        self.generative_model = generative_model
        self.use_hybrid = use_hybrid
        self._prompt_template = hyde_prompt_template or self.DEFAULT_PROMPT

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _expand_query(self, query: str) -> str:
        """Call the generative model to produce a hypothetical document."""
        prompt = self._prompt_template.format(query=query)
        response = self.generative_model.generate_content(prompt)
        expanded = response.text.strip()
        logger.debug("[Strategy B] Expanded %r → %r", query, expanded[:80])
        return expanded

    def _build_search_vector(self, query: str) -> np.ndarray:
        """
        Build the final search vector.

        * Pure HyDE:   embed(hypothetical_doc)
        * Hybrid HyDE: (embed(query) + embed(hypothetical_doc)) / 2, then renormalise
        """
        hypothetical_doc = self._expand_query(query)
        hyde_vec = self.embedding_model.embed_query(hypothetical_doc)

        if not self.use_hybrid:
            return hyde_vec

        # Hybrid: average raw query and HyDE vectors, then re-normalise
        raw_vec = self.embedding_model.embed_query(query)
        avg_vec = (raw_vec + hyde_vec) / 2.0
        norm = np.linalg.norm(avg_vec)
        if norm > 0:
            avg_vec = avg_vec / norm
        return avg_vec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """
        Generate a hypothetical document, embed it, and search.

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
        search_vec = self._build_search_vector(query)
        results = self.vector_store.search(search_vec, top_k=top_k)
        logger.info(
            "[Strategy B] Retrieved %d results for query: %r (hybrid=%s)",
            len(results), query, self.use_hybrid,
        )
        return results

    def get_expansion(self, query: str) -> str:
        """Expose the hypothetical document for logging / inspection."""
        return self._expand_query(query)
