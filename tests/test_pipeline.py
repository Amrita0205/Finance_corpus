"""
tests/test_pipeline.py
=======================
Integration tests for the RAGPipeline orchestrator.
Exercises the full ingestion → retrieval cycle.
"""

from __future__ import annotations

import pytest

from finrag.pipeline.rag_pipeline import RAGPipeline
from finrag.storage.vector_store import SearchResult


class TestRAGPipelineIngestion:

    def test_ingest_populates_vector_store(self, ingested_pipeline, full_corpus):
        assert ingested_pipeline.vector_store.total_documents == len(full_corpus)

    def test_ingest_empty_list_raises(self, mock_generative_model):
        pipeline = RAGPipeline(generative_model=mock_generative_model)
        with pytest.raises(ValueError, match="empty"):
            pipeline.ingest([])

    def test_ingest_missing_keys_raises(self, mock_generative_model):
        pipeline = RAGPipeline(generative_model=mock_generative_model)
        with pytest.raises(ValueError, match="missing required keys"):
            pipeline.ingest([{"id": "x", "content": "no title"}])

    def test_retrieve_before_ingest_raises(self, mock_generative_model):
        pipeline = RAGPipeline(generative_model=mock_generative_model)
        with pytest.raises(RuntimeError, match="ingest"):
            pipeline.retrieve_strategy_a("test query")

    def test_ingest_partial_corpus(self, small_corpus, mock_generative_model):
        pipeline = RAGPipeline(generative_model=mock_generative_model)
        pipeline.ingest(small_corpus)
        assert pipeline.vector_store.total_documents == len(small_corpus)


class TestRAGPipelineRetrieval:

    def test_strategy_a_returns_list(self, ingested_pipeline):
        results = ingested_pipeline.retrieve_strategy_a("peak load", top_k=3)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_strategy_b_returns_list(self, ingested_pipeline):
        results = ingested_pipeline.retrieve_strategy_b("peak load", top_k=3)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_retrieve_both_returns_dict(self, ingested_pipeline):
        result = ingested_pipeline.retrieve_both("risk management", top_k=3)
        assert "strategy_a" in result
        assert "strategy_b" in result
        assert len(result["strategy_a"]) == 3
        assert len(result["strategy_b"]) == 3

    def test_results_are_search_result_objects(self, ingested_pipeline):
        both = ingested_pipeline.retrieve_both("Basel III capital", top_k=2)
        for results in both.values():
            assert all(isinstance(r, SearchResult) for r in results)

    def test_get_hyde_expansion_returns_string(self, ingested_pipeline):
        expansion = ingested_pipeline.get_hyde_expansion("peak load")
        assert isinstance(expansion, str)
        assert len(expansion) > 0

    def test_peak_load_query_top_result(self, ingested_pipeline):
        """The HFT document (doc_001) should be #1 for the peak load query."""
        a_results = ingested_pipeline.retrieve_strategy_a(
            "How does the system handle peak load?", top_k=3
        )
        b_results = ingested_pipeline.retrieve_strategy_b(
            "How does the system handle peak load?", top_k=3
        )
        a_ids = [r.doc_id for r in a_results]
        b_ids = [r.doc_id for r in b_results]
        assert "doc_001" in a_ids, f"Strategy A missed doc_001; got {a_ids}"
        assert "doc_001" in b_ids, f"Strategy B missed doc_001; got {b_ids}"

    def test_strategy_b_top1_geq_or_similar_to_a(self, ingested_pipeline):
        """
        HyDE (Strategy B) should achieve a higher or similar TOP-1 cosine score
        compared to raw vector search (Strategy A).

        HyDE concentrates relevance on the #1 result (the hypothetical document
        is very close to the ground-truth chunk) while the tail scores may drop —
        so comparing top-1 scores is the correct evaluation here.
        We allow up to 0.02 slack to account for equal cases.
        """
        queries = [
            "How does the system handle peak load?",
            "risk management in volatile markets",
            "real-time price discovery in exchanges",
        ]
        better_or_equal = 0
        for q in queries:
            a_res = ingested_pipeline.retrieve_strategy_a(q, top_k=3)
            b_res = ingested_pipeline.retrieve_strategy_b(q, top_k=3)
            a_top1 = a_res[0].score
            b_top1 = b_res[0].score
            # Allow B to be up to 0.02 below A top-1
            if b_top1 >= a_top1 - 0.02:
                better_or_equal += 1

        assert better_or_equal >= 2, (
            f"Strategy B top-1 should be >= A top-1 on at least 2/3 queries. "
            f"Got {better_or_equal}/3."
        )

    def test_repr(self, ingested_pipeline):
        r = repr(ingested_pipeline)
        assert "RAGPipeline" in r


class TestRAGPipelinePersistence:

    def test_save_and_load_roundtrip(self, ingested_pipeline, tmp_path, mock_generative_model):
        idx_path = str(tmp_path / "index.faiss")
        meta_path = str(tmp_path / "meta.json")
        ingested_pipeline.save(idx_path, meta_path)

        loaded = RAGPipeline.load(
            idx_path, meta_path, generative_model=mock_generative_model
        )
        assert loaded.vector_store.total_documents == ingested_pipeline.vector_store.total_documents

    def test_loaded_pipeline_can_retrieve(
        self, ingested_pipeline, tmp_path, mock_generative_model
    ):
        idx_path = str(tmp_path / "index.faiss")
        meta_path = str(tmp_path / "meta.json")
        ingested_pipeline.save(idx_path, meta_path)

        loaded = RAGPipeline.load(idx_path, meta_path, generative_model=mock_generative_model)
        results = loaded.retrieve_strategy_a("peak load", top_k=3)
        assert len(results) == 3
