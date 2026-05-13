"""
tests/conftest.py
==================
Shared pytest fixtures used across all test modules.

Fixture design
--------------
* ``mock_modules``  — patches sys.modules so that ``import vertexai`` resolves
  to our mock classes. Applied as a session-scoped autouse fixture.
* ``embedding_model`` — a real EmbeddingModel backed by sentence-transformers.
* ``sample_corpus``   — the 10 finance documents (subset for fast tests).
* ``ingested_pipeline`` — a fully ingested RAGPipeline ready for retrieval tests.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest

from finrag.mocks.vertexai_mocks import (
    MockGenerativeModel,
    MockTextEmbeddingModel,
    mock_vertexai_module,
)


# ---------------------------------------------------------------------------
# Session-level: inject mock vertexai into sys.modules
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def mock_vertexai_sdk():
    """
    Patch sys.modules to provide the mock Vertex AI SDK for the entire test
    session. This prevents ImportError when vertexai is not installed and
    ensures all tests are hermetic (no GCP credentials required).
    """
    with patch.dict(sys.modules, mock_vertexai_module()):
        yield


# ---------------------------------------------------------------------------
# Shared corpus fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def full_corpus():
    from data.finance_corpus import FINANCE_CORPUS
    return FINANCE_CORPUS


@pytest.fixture(scope="session")
def small_corpus(full_corpus):
    """First 3 documents — fast for unit tests."""
    return full_corpus[:3]


# ---------------------------------------------------------------------------
# Embedding model fixture (shared; expensive to re-load)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def embedding_model():
    from finrag.embeddings.embedding_model import EmbeddingModel
    return EmbeddingModel(normalise=True)


# ---------------------------------------------------------------------------
# Mock model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mock_text_embedding_model():
    return MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")


@pytest.fixture(scope="session")
def mock_generative_model():
    return MockGenerativeModel("gemini-1.0-pro")


# ---------------------------------------------------------------------------
# Vector store fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_vector_store():
    from finrag.storage.vector_store import FAISSVectorStore
    return FAISSVectorStore()


@pytest.fixture(scope="session")
def populated_vector_store(embedding_model, full_corpus):
    """
    A FAISSVectorStore pre-loaded with all 10 finance documents.
    Session-scoped because embedding is slow.
    """
    from finrag.storage.vector_store import FAISSVectorStore

    texts = [doc["content"] for doc in full_corpus]
    embeddings = embedding_model.embed_documents(texts)
    metadata = [
        {"id": doc["id"], "title": doc["title"], "content": doc["content"]}
        for doc in full_corpus
    ]
    store = FAISSVectorStore()
    store.add_documents(embeddings, metadata)
    return store


# ---------------------------------------------------------------------------
# Full pipeline fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ingested_pipeline(full_corpus, mock_generative_model):
    """
    A fully ingested RAGPipeline — expensive to build, so session-scoped.
    Uses the real sentence-transformers encoder but a mock generative model.
    """
    from finrag.pipeline.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(generative_model=mock_generative_model)
    pipeline.ingest(full_corpus)
    return pipeline


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def random_unit_vector():
    """Returns a random L2-normalised 384-dim vector (all-MiniLM-L6-v2 dim)."""
    v = np.random.randn(384).astype(np.float32)
    return v / np.linalg.norm(v)
