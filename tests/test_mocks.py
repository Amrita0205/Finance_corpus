"""
tests/test_mocks.py
====================
Tests for MockTextEmbeddingModel and MockGenerativeModel.
Verifies that the mocks faithfully mirror the Vertex AI SDK interfaces.
"""

from __future__ import annotations

import numpy as np
import pytest

from finrag.mocks.vertexai_mocks import (
    MockGenerativeModel,
    MockGenerativeResponse,
    MockTextEmbedding,
    MockTextEmbeddingInstance,
    MockTextEmbeddingModel,
    mock_vertexai_module,
)


# ---------------------------------------------------------------------------
# MockTextEmbeddingModel
# ---------------------------------------------------------------------------

class TestMockTextEmbeddingModel:

    def test_from_pretrained_returns_instance(self):
        model = MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        assert isinstance(model, MockTextEmbeddingModel)

    def test_get_embeddings_returns_list(self, mock_text_embedding_model):
        results = mock_text_embedding_model.get_embeddings(["hello world"])
        assert isinstance(results, list)
        assert len(results) == 1

    def test_get_embeddings_returns_mock_embedding(self, mock_text_embedding_model):
        results = mock_text_embedding_model.get_embeddings(["test"])
        assert isinstance(results[0], MockTextEmbedding)

    def test_embedding_values_is_list_of_floats(self, mock_text_embedding_model):
        results = mock_text_embedding_model.get_embeddings(["quantitative finance"])
        values = results[0].values
        assert isinstance(values, list)
        assert all(isinstance(v, float) for v in values)

    def test_embedding_dimension_consistent(self, mock_text_embedding_model):
        results = mock_text_embedding_model.get_embeddings(["a", "b", "c"])
        dims = [len(r.values) for r in results]
        assert len(set(dims)) == 1, "All embeddings must have the same dimension"

    def test_batch_size_matches_input(self, mock_text_embedding_model):
        texts = ["risk management", "VaR", "Black-Scholes", "HFT"]
        results = mock_text_embedding_model.get_embeddings(texts)
        assert len(results) == len(texts)

    def test_accepts_text_embedding_instance(self, mock_text_embedding_model):
        instances = [MockTextEmbeddingInstance(text="options pricing", task_type="RETRIEVAL_QUERY")]
        results = mock_text_embedding_model.get_embeddings(instances)
        assert len(results) == 1

    def test_different_texts_produce_different_embeddings(self, mock_text_embedding_model):
        results = mock_text_embedding_model.get_embeddings(["credit risk", "high frequency trading"])
        arr0 = np.array(results[0].values)
        arr1 = np.array(results[1].values)
        cosine = np.dot(arr0, arr1) / (np.linalg.norm(arr0) * np.linalg.norm(arr1))
        assert cosine < 0.999, "Distinct texts should not yield identical embeddings"

    def test_same_text_produces_deterministic_embedding(self, mock_text_embedding_model):
        r1 = mock_text_embedding_model.get_embeddings(["Basel III capital requirements"])
        r2 = mock_text_embedding_model.get_embeddings(["Basel III capital requirements"])
        np.testing.assert_array_almost_equal(r1[0].values, r2[0].values)


# ---------------------------------------------------------------------------
# MockGenerativeModel
# ---------------------------------------------------------------------------

class TestMockGenerativeModel:

    def test_from_pretrained_returns_instance(self):
        model = MockGenerativeModel.from_pretrained("gemini-1.0-pro")
        assert isinstance(model, MockGenerativeModel)

    def test_generate_content_returns_response(self, mock_generative_model):
        response = mock_generative_model.generate_content("What is peak load?")
        assert isinstance(response, MockGenerativeResponse)

    def test_response_has_text_attribute(self, mock_generative_model):
        response = mock_generative_model.generate_content("peak load handling in trading")
        assert hasattr(response, "text")
        assert isinstance(response.text, str)
        assert len(response.text) > 0

    def test_finance_expansion_triggers(self, mock_generative_model):
        """Known trigger phrases should produce domain-specific expansions."""
        response = mock_generative_model.generate_content(
            "How does the system handle peak load?"
        )
        text_lower = response.text.lower()
        # Should contain HFT / trading-system terminology
        assert any(kw in text_lower for kw in ["trading", "order", "load", "latency", "market"])

    def test_risk_expansion_triggers(self, mock_generative_model):
        response = mock_generative_model.generate_content("risk management strategies")
        text_lower = response.text.lower()
        assert any(kw in text_lower for kw in ["var", "risk", "portfolio", "monte carlo", "stress"])

    def test_generic_fallback_for_unknown_query(self, mock_generative_model):
        response = mock_generative_model.generate_content("xyzzyplutonium widget")
        assert len(response.text) > 10  # falls back gracefully

    def test_expansion_is_longer_than_query(self, mock_generative_model):
        query = "peak load"
        response = mock_generative_model.generate_content(query)
        assert len(response.text) > len(query)

    def test_send_message_equivalent(self, mock_generative_model):
        """start_chat + send_message should work (multi-turn stub)."""
        chat = mock_generative_model.start_chat()
        response = chat.send_message("price discovery")
        assert isinstance(response, MockGenerativeResponse)


# ---------------------------------------------------------------------------
# mock_vertexai_module
# ---------------------------------------------------------------------------

class TestMockVertexAIModule:

    def test_returns_dict(self):
        result = mock_vertexai_module()
        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        result = mock_vertexai_module()
        assert "vertexai" in result
        assert "vertexai.language_models" in result
        assert "vertexai.generative_models" in result

    def test_language_models_has_text_embedding_model(self):
        mods = mock_vertexai_module()
        lm = mods["vertexai.language_models"]
        assert hasattr(lm, "TextEmbeddingModel")

    def test_generative_models_has_generative_model(self):
        mods = mock_vertexai_module()
        gm = mods["vertexai.generative_models"]
        assert hasattr(gm, "GenerativeModel")
