"""
tests/test_vector_store.py
===========================
Unit tests for FAISSVectorStore.
"""

from __future__ import annotations

import numpy as np
import pytest

from finrag.storage.vector_store import FAISSVectorStore, SearchResult


def _make_unit_vector(dim: int = 384, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_batch(n: int = 5, dim: int = 384) -> tuple[np.ndarray, list[dict]]:
    vecs = np.array([_make_unit_vector(dim, seed=i) for i in range(n)], dtype=np.float32)
    meta = [{"id": f"doc_{i:03d}", "title": f"Document {i}", "content": f"Content {i}"} for i in range(n)]
    return vecs, meta


class TestFAISSVectorStore:

    def test_add_and_total_documents(self, empty_vector_store):
        vecs, meta = _make_batch(3)
        empty_vector_store.add_documents(vecs, meta)
        assert empty_vector_store.total_documents == 3

    def test_search_returns_list_of_search_results(self, populated_vector_store, random_unit_vector):
        results = populated_vector_store.search(random_unit_vector, top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_returns_correct_top_k(self, populated_vector_store, random_unit_vector):
        for k in (1, 3, 5):
            results = populated_vector_store.search(random_unit_vector, top_k=k)
            assert len(results) == k

    def test_search_scores_descending(self, populated_vector_store, random_unit_vector):
        results = populated_vector_store.search(random_unit_vector, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "Results must be in descending score order"

    def test_exact_match_has_highest_score(self, embedding_model, full_corpus):
        """A query vector identical to a stored vector should rank #1 with score ≈ 1."""
        store = FAISSVectorStore()
        texts = [d["content"] for d in full_corpus[:3]]
        embs = embedding_model.embed_documents(texts)
        meta = [{"id": d["id"], "title": d["title"], "content": d["content"]} for d in full_corpus[:3]]
        store.add_documents(embs, meta)

        query_vec = embs[1]  # same as doc at position 1
        results = store.search(query_vec, top_k=1)
        assert results[0].doc_id == full_corpus[1]["id"]
        assert results[0].score > 0.99

    def test_ranks_are_sequential(self, populated_vector_store, random_unit_vector):
        results = populated_vector_store.search(random_unit_vector, top_k=5)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_result_fields_populated(self, populated_vector_store, random_unit_vector):
        results = populated_vector_store.search(random_unit_vector, top_k=1)
        r = results[0]
        assert r.doc_id
        assert r.title
        assert r.content
        assert isinstance(r.score, float)

    def test_search_on_empty_store_raises(self, empty_vector_store, random_unit_vector):
        with pytest.raises(ValueError, match="empty"):
            empty_vector_store.search(random_unit_vector)

    def test_mismatched_embeddings_metadata_raises(self, empty_vector_store):
        vecs, meta = _make_batch(3)
        with pytest.raises(AssertionError):
            empty_vector_store.add_documents(vecs, meta[:2])  # meta too short

    def test_filter_ids(self, populated_vector_store, embedding_model, full_corpus):
        query_vec = embedding_model.embed_query("high frequency trading")
        allowed = [full_corpus[0]["id"], full_corpus[2]["id"]]
        results = populated_vector_store.search(query_vec, top_k=3, filter_ids=allowed)
        for r in results:
            assert r.doc_id in allowed

    def test_clear_resets_store(self, empty_vector_store):
        vecs, meta = _make_batch(4)
        empty_vector_store.add_documents(vecs, meta)
        assert empty_vector_store.total_documents == 4
        empty_vector_store.clear()
        assert empty_vector_store.total_documents == 0

    def test_repr_string(self, populated_vector_store):
        r = repr(populated_vector_store)
        assert "FAISSVectorStore" in r
        assert str(populated_vector_store.total_documents) in r

    def test_incremental_add(self, empty_vector_store):
        v1, m1 = _make_batch(2)
        v2, m2 = _make_batch(3, dim=384)
        # Use different seeds to avoid ID collision in meta
        m2 = [{"id": f"extra_{i}", "title": f"Extra {i}", "content": f"Extra content {i}"} for i in range(3)]
        empty_vector_store.add_documents(v1, m1)
        empty_vector_store.add_documents(v2, m2)
        assert empty_vector_store.total_documents == 5