"""Tests for vector_store module."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.vector_store import VectorStore
from src.text_splitter import TextChunk


@pytest.fixture
def sample_chunks():
    """Create sample TextChunk objects for testing."""
    return [
        TextChunk(
            content="MACD is a trend-following momentum indicator",
            metadata={"source": "technical_indicators.md", "page": 1},
            chunk_id="chunk_001",
            start_idx=0,
            end_idx=45,
        ),
        TextChunk(
            content="RSI measures the speed of price changes",
            metadata={"source": "technical_indicators.md", "page": 2},
            chunk_id="chunk_002",
            start_idx=46,
            end_idx=85,
        ),
        TextChunk(
            content="Bollinger Bands show volatility",
            metadata={"source": "technical_indicators.md", "page": 3},
            chunk_id="chunk_003",
            start_idx=86,
            end_idx=117,
        ),
    ]


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings."""
    return [[0.1] * 10, [0.2] * 10, [0.3] * 10]


@pytest.fixture
def tmp_persist_dir(tmp_path):
    """Create a temporary persist directory."""
    return str(tmp_path / "test_chroma")


class TestVectorStoreInit:
    """Test VectorStore initialization."""

    def test_default_init(self, tmp_persist_dir):
        """Test default initialization."""
        store = VectorStore(persist_dir=tmp_persist_dir)
        assert store.get_count() == 0

    def test_custom_collection(self, tmp_persist_dir):
        """Test custom collection name."""
        store = VectorStore(
            persist_dir=tmp_persist_dir, collection_name="custom_kb"
        )
        assert store.get_count() == 0


class TestVectorStoreAddChunks:
    """Test adding chunks to the store."""

    def test_add_chunks_returns_count(self, tmp_persist_dir, sample_chunks, sample_embeddings):
        """Test add_chunks returns the number of added chunks."""
        store = VectorStore(persist_dir=tmp_persist_dir)
        count = store.add_chunks(sample_chunks, sample_embeddings)
        assert count == 3

    def test_add_chunks_increments_count(self, tmp_persist_dir, sample_chunks, sample_embeddings):
        """Test that adding chunks increments the count."""
        store = VectorStore(persist_dir=tmp_persist_dir)
        store.add_chunks(sample_chunks[:2], sample_embeddings[:2])
        assert store.get_count() == 2

        store.add_chunks(sample_chunks[2:], sample_embeddings[2:])
        assert store.get_count() == 3

    def test_add_empty_chunks(self, tmp_persist_dir):
        """Test adding empty list."""
        store = VectorStore(persist_dir=tmp_persist_dir)
        count = store.add_chunks([], [])
        assert count == 0


class TestVectorStoreSearch:
    """Test searching the store."""

    def test_search_returns_results(self, tmp_persist_dir, sample_chunks, sample_embeddings):
        """Test basic search returns results."""
        store = VectorStore(persist_dir=tmp_persist_dir)
        store.add_chunks(sample_chunks, sample_embeddings)

        results = store.search(query_embedding=[0.1] * 10, top_k=3)
        assert len(results) > 0
        assert all("content" in r for r in results)
        assert all("metadata" in r for r in results)
        assert all("distance" in r for r in results)

    def test_search_top_k(self, tmp_persist_dir, sample_chunks, sample_embeddings):
        """Test search respects top_k."""
        store = VectorStore(persist_dir=tmp_persist_dir)
        store.add_chunks(sample_chunks, sample_embeddings)

        results = store.search(query_embedding=[0.1] * 10, top_k=2)
        assert len(results) <= 2

    def test_search_empty_store(self, tmp_persist_dir):
        """Test searching empty store."""
        store = VectorStore(persist_dir=tmp_persist_dir)
        results = store.search(query_embedding=[0.1] * 10, top_k=5)
        assert results == []


class TestVectorStoreDelete:
    """Test deleting from the store."""

    def test_delete_collection(self, tmp_persist_dir, sample_chunks, sample_embeddings):
        """Test deleting the collection."""
        store = VectorStore(persist_dir=tmp_persist_dir)
        store.add_chunks(sample_chunks, sample_embeddings)
        assert store.get_count() == 3

        store.delete_collection()
        assert store.get_count() == 0


class TestVectorStorePersist:
    """Test persistence."""

    def test_persist(self, tmp_persist_dir, sample_chunks, sample_embeddings):
        """Test that persist doesn't raise."""
        store = VectorStore(persist_dir=tmp_persist_dir)
        store.add_chunks(sample_chunks, sample_embeddings)
        store.persist()  # Should not raise
