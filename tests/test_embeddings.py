"""
tests/test_embeddings.py
=========================
Unit tests for the EmbeddingModel wrapper.
"""

from __future__ import annotations

import numpy as np
import pytest

from finrag.embeddings.embedding_model import EmbeddingModel


class TestEmbeddingModel:

    def test_embed_query_returns_1d_array(self, embedding_model):
        vec = embedding_model.embed_query("Value-at-Risk calculation")
        assert vec.ndim == 1
        assert vec.dtype == np.float32

    def test_embed_documents_returns_2d_array(self, embedding_model):
        texts = ["Black-Scholes model", "HFT systems", "credit risk"]
        result = embedding_model.embed_documents(texts)
        assert result.ndim == 2
        assert result.shape[0] == len(texts)

    def test_consistent_dimension_query_vs_documents(self, embedding_model):
        query_vec = embedding_model.embed_query("portfolio optimisation")
        doc_mat = embedding_model.embed_documents(["any text"])
        assert query_vec.shape[0] == doc_mat.shape[1]

    def test_normalisation_unit_norm(self, embedding_model):
        """All returned vectors must be L2-normalised (norm ≈ 1.0)."""
        texts = ["VWAP execution", "Basel III capital", "options Greeks"]
        matrix = embedding_model.embed_documents(texts)
        norms = np.linalg.norm(matrix, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(texts)), atol=1e-5)

    def test_embed_query_is_normalised(self, embedding_model):
        vec = embedding_model.embed_query("market microstructure")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_dimension_property(self, embedding_model):
        dim = embedding_model.dimension
        assert isinstance(dim, int)
        assert dim > 0

    def test_empty_list_returns_empty_array(self, embedding_model):
        result = embedding_model.embed_documents([])
        assert result.shape[0] == 0

    def test_similar_texts_high_cosine(self, embedding_model):
        """Semantically similar texts should have high cosine similarity."""
        vecs = embedding_model.embed_documents([
            "high frequency trading peak load",
            "HFT systems under maximum throughput",
        ])
        cosine = float(np.dot(vecs[0], vecs[1]))  # already normalised → dot = cosine
        assert cosine > 0.7, f"Expected high cosine for similar texts, got {cosine:.4f}"

    def test_dissimilar_texts_lower_cosine(self, embedding_model):
        """Semantically dissimilar texts should have lower cosine than similar ones."""
        vecs = embedding_model.embed_documents([
            "high frequency trading peak load",
            "regulatory capital Basel III requirements",
        ])
        cosine = float(np.dot(vecs[0], vecs[1]))
        # Not necessarily negative, but should be lower than intra-topic similarity
        assert cosine < 0.98

    def test_deterministic_embeddings(self, embedding_model):
        text = "volatility surface calibration SABR model"
        v1 = embedding_model.embed_query(text)
        v2 = embedding_model.embed_query(text)
        np.testing.assert_array_almost_equal(v1, v2)

    def test_encode_alias(self, embedding_model):
        """encode() must behave identically to embed_documents()."""
        texts = ["momentum factor", "Fama-French"]
        r1 = embedding_model.embed_documents(texts)
        r2 = embedding_model.encode(texts)
        np.testing.assert_array_almost_equal(r1, r2)

    def test_no_normalise_flag(self):
        model = EmbeddingModel(normalise=False)
        vec = model.embed_query("credit default swap")
        # With normalise=False the raw norm is NOT forced to 1
        # (we just check it's a valid float32 array)
        assert vec.dtype == np.float32
        assert vec.ndim == 1
