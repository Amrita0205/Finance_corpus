"""
tests/test_retrieval.py
========================
Tests for Strategy A (RawVectorRetriever) and Strategy B (HyDERetriever).
Uses unittest.mock to verify that the generative model is called correctly
and that the pipeline produces plausible rankings.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from finrag.retrieval.strategy_a import RawVectorRetriever
from finrag.retrieval.strategy_b import HyDERetriever
from finrag.storage.vector_store import SearchResult
from finrag.mocks.vertexai_mocks import MockGenerativeModel, MockGenerativeResponse


# ---------------------------------------------------------------------------
# Strategy A
# ---------------------------------------------------------------------------

class TestRawVectorRetriever:

    def test_strategy_name(self, populated_vector_store, embedding_model):
        retriever = RawVectorRetriever(populated_vector_store, embedding_model)
        assert "Strategy A" in retriever.strategy_name

    def test_retrieve_returns_list(self, populated_vector_store, embedding_model):
        retriever = RawVectorRetriever(populated_vector_store, embedding_model)
        results = retriever.retrieve("peak load handling", top_k=3)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_retrieve_returns_search_results(self, populated_vector_store, embedding_model):
        retriever = RawVectorRetriever(populated_vector_store, embedding_model)
        results = retriever.retrieve("options pricing", top_k=2)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_hft_query_retrieves_relevant_doc(self, populated_vector_store, embedding_model):
        """'peak load' query should surface the HFT document (doc_001) in top 3."""
        retriever = RawVectorRetriever(populated_vector_store, embedding_model)
        results = retriever.retrieve("How does the system handle peak load?", top_k=3)
        doc_ids = [r.doc_id for r in results]
        assert "doc_001" in doc_ids, (
            f"Expected doc_001 (HFT) in top-3 for peak load query. Got: {doc_ids}"
        )

    def test_risk_query_retrieves_risk_doc(self, populated_vector_store, embedding_model):
        retriever = RawVectorRetriever(populated_vector_store, embedding_model)
        results = retriever.retrieve("Value-at-Risk and portfolio risk management", top_k=3)
        doc_ids = [r.doc_id for r in results]
        assert "doc_002" in doc_ids, f"Expected doc_002 (VaR/risk) in top-3. Got: {doc_ids}"

    def test_scores_descending(self, populated_vector_store, embedding_model):
        retriever = RawVectorRetriever(populated_vector_store, embedding_model)
        results = retriever.retrieve("Black-Scholes options model", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_repr(self, populated_vector_store, embedding_model):
        retriever = RawVectorRetriever(populated_vector_store, embedding_model)
        assert "RawVectorRetriever" in repr(retriever)

    def test_embedding_model_called(self, populated_vector_store, embedding_model):
        """Verify embed_query is invoked (white-box)."""
        retriever = RawVectorRetriever(populated_vector_store, embedding_model)
        with patch.object(embedding_model, "embed_query", wraps=embedding_model.embed_query) as mock_eq:
            retriever.retrieve("liquidity premium", top_k=2)
            mock_eq.assert_called_once_with("liquidity premium")


# ---------------------------------------------------------------------------
# Strategy B
# ---------------------------------------------------------------------------

class TestHyDERetriever:

    @pytest.fixture
    def hyde_retriever(self, populated_vector_store, embedding_model, mock_generative_model):
        return HyDERetriever(
            vector_store=populated_vector_store,
            embedding_model=embedding_model,
            generative_model=mock_generative_model,
            use_hybrid=False,
        )

    @pytest.fixture
    def hybrid_retriever(self, populated_vector_store, embedding_model, mock_generative_model):
        return HyDERetriever(
            vector_store=populated_vector_store,
            embedding_model=embedding_model,
            generative_model=mock_generative_model,
            use_hybrid=True,
        )

    def test_strategy_name(self, hyde_retriever):
        assert "Strategy B" in hyde_retriever.strategy_name

    def test_retrieve_returns_list(self, hyde_retriever):
        results = hyde_retriever.retrieve("peak load handling", top_k=3)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_retrieve_returns_search_results(self, hyde_retriever):
        results = hyde_retriever.retrieve("volatility surface calibration", top_k=2)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_generative_model_called_once_per_retrieve(
        self, populated_vector_store, embedding_model
    ):
        """The generative model must be called exactly once per retrieve() call."""
        mock_gen = MagicMock()
        mock_gen.generate_content.return_value = MockGenerativeResponse(
            text="High-frequency trading systems handle peak load through autoscaling."
        )
        retriever = HyDERetriever(populated_vector_store, embedding_model, mock_gen)
        retriever.retrieve("peak load", top_k=3)
        mock_gen.generate_content.assert_called_once()

    def test_generate_content_receives_query_in_prompt(
        self, populated_vector_store, embedding_model
    ):
        mock_gen = MagicMock()
        mock_gen.generate_content.return_value = MockGenerativeResponse(text="some expansion")
        retriever = HyDERetriever(populated_vector_store, embedding_model, mock_gen)
        query = "What is the settlement cycle?"
        retriever.retrieve(query, top_k=1)
        prompt_used = mock_gen.generate_content.call_args[0][0]
        assert query in prompt_used

    def test_hyde_embed_uses_expansion_not_raw_query(
        self, populated_vector_store, embedding_model
    ):
        """embed_query must be called with the expanded text, NOT the raw query."""
        expansion = "This is the hypothetical expanded finance document about settlement."
        mock_gen = MagicMock()
        mock_gen.generate_content.return_value = MockGenerativeResponse(text=expansion)

        retriever = HyDERetriever(populated_vector_store, embedding_model, mock_gen)
        with patch.object(embedding_model, "embed_query", wraps=embedding_model.embed_query) as mock_eq:
            retriever.retrieve("settlement cycle", top_k=1)
            # The argument to embed_query should be the expansion, not the original query
            called_with = mock_eq.call_args[0][0]
            assert called_with == expansion.strip()

    def test_hybrid_mode_averages_vectors(
        self, populated_vector_store, embedding_model, mock_generative_model
    ):
        """Hybrid mode must call embed_query twice (raw + expansion)."""
        retriever = HyDERetriever(
            populated_vector_store, embedding_model, mock_generative_model, use_hybrid=True
        )
        with patch.object(embedding_model, "embed_query", wraps=embedding_model.embed_query) as mock_eq:
            retriever.retrieve("credit default swap valuation", top_k=1)
            assert mock_eq.call_count == 2, "Hybrid mode must embed both raw query and HyDE expansion"

    def test_get_expansion_returns_string(self, hyde_retriever):
        text = hyde_retriever.get_expansion("peak load capacity")
        assert isinstance(text, str)
        assert len(text) > 10

    def test_hft_query_retrieves_relevant_doc(self, hyde_retriever):
        results = hyde_retriever.retrieve("How does the system handle peak load?", top_k=3)
        doc_ids = [r.doc_id for r in results]
        assert "doc_001" in doc_ids, (
            f"Expected doc_001 (HFT) in top-3 for peak load query via HyDE. Got: {doc_ids}"
        )

    def test_scores_descending(self, hyde_retriever):
        results = hyde_retriever.retrieve("options pricing Black-Scholes", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_custom_prompt_template(self, populated_vector_store, embedding_model):
        mock_gen = MagicMock()
        mock_gen.generate_content.return_value = MockGenerativeResponse(text="custom expansion")
        custom_template = "Custom: {query}"
        retriever = HyDERetriever(
            populated_vector_store, embedding_model, mock_gen,
            hyde_prompt_template=custom_template
        )
        retriever.retrieve("some query", top_k=1)
        prompt_used = mock_gen.generate_content.call_args[0][0]
        assert prompt_used == "Custom: some query"
