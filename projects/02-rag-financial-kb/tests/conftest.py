"""Shared pytest fixtures and configuration."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

# 确保项目根目录和 src/ 目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client that returns predictable responses."""
    client = MagicMock()
    # Chat completion mock
    chat_response = MagicMock()
    chat_response.choices = [MagicMock()]
    chat_response.choices[0].message = MagicMock()
    chat_response.choices[0].message.content = "Mocked answer from OpenAI"
    client.chat.completions.create.return_value = chat_response
    return client


@pytest.fixture
def mock_embedding_model():
    """Mock EmbeddingModel / LocalEmbeddingModel instance."""
    model = MagicMock()
    model.embed_texts.return_value = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
    model.embed_query.return_value = [0.15] * 1536
    model.get_dimension.return_value = 1536
    return model


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore instance."""
    store = MagicMock()
    store.search.return_value = [
        {
            "content": "Test document content about financial risk management.",
            "metadata": {"source": "test.pdf", "page": 1, "title": "Risk"},
            "distance": 0.05,
        },
        {
            "content": "Portfolio diversification reduces unsystematic risk.",
            "metadata": {"source": "test.pdf", "page": 2, "title": "Diversification"},
            "distance": 0.13,
        },
    ]
    store.add_chunks.return_value = 2
    store.get_count.return_value = 2
    store._collection_name = "financial_kb"
    store._persist_dir = "/tmp/test_chroma"
    return store


@pytest.fixture
def mock_retriever(mock_vector_store, mock_embedding_model):
    """Mock Retriever instance with similarity_search."""
    retriever = MagicMock()
    retriever.similarity_search.return_value = [
        {
            "content": "Test document content about financial risk management.",
            "metadata": {"source": "test.pdf", "page": 1, "title": "Risk"},
            "distance": 0.05,
        },
        {
            "content": "Portfolio diversification reduces unsystematic risk.",
            "metadata": {"source": "test.pdf", "page": 2, "title": "Diversification"},
            "distance": 0.13,
        },
    ]
    return retriever


@pytest.fixture
def sample_text():
    """Sample long text for splitting tests."""
    return (
        "Financial risk management is essential for any investment portfolio. "
        "Diversification helps mitigate specific risks in investments. "
        "Technical analysis uses historical price data for predictions. "
        "Fundamental analysis examines financial statements and economic indicators. "
        "Value investing focuses on stocks that appear underpriced by fundamental analysis. "
        "Momentum investing relies on the belief that existing trends will continue. "
        "Portfolio rebalancing ensures the asset allocation remains aligned with goals. "
        "Risk-adjusted returns measure the return earned per unit of risk taken. "
        "The Sharpe ratio is a common measure of risk-adjusted performance. "
        "Maximum drawdown measures the largest peak-to-trough decline in portfolio value."
    )


@pytest.fixture
def sample_documents():
    """Sample Document instances for testing."""
    from src.document_loader import Document

    return [
        Document(
            content="Financial risk management is crucial for portfolio stability.",
            metadata={
                "source": "test1.txt",
                "page": None,
                "title": "Risk Management",
                "doc_type": "text",
            },
        ),
        Document(
            content="Diversification helps mitigate specific risks in investments.",
            metadata={
                "source": "test2.txt",
                "page": None,
                "title": "Diversification",
                "doc_type": "text",
            },
        ),
        Document(
            content="Technical analysis uses historical price data for predictions.",
            metadata={
                "source": "test3.md",
                "page": None,
                "title": "Technical Analysis",
                "doc_type": "markdown",
            },
        ),
    ]


@pytest.fixture
def sample_chunks():
    """Sample TextChunk instances for testing."""
    from src.text_splitter import TextChunk

    return [
        TextChunk(
            content="Financial risk management is crucial.",
            metadata={"source": "test1.txt", "title": "Risk"},
            chunk_id="test1.txt::chunk_0_abc12345",
            start_idx=0,
            end_idx=36,
        ),
        TextChunk(
            content="Diversification helps mitigate risks.",
            metadata={"source": "test2.txt", "title": "Diversification"},
            chunk_id="test2.txt::chunk_1_def67890",
            start_idx=0,
            end_idx=36,
        ),
        TextChunk(
            content="Technical analysis uses historical data.",
            metadata={"source": "test3.md", "title": "TA"},
            chunk_id="test3.md::chunk_2_ghi11111",
            start_idx=0,
            end_idx=38,
        ),
    ]
