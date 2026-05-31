"""Tests for src/text_splitter module."""

import pytest

from src.text_splitter import (
    TextChunk,
    recursive_split,
    semantic_split,
    financial_split,
    split_documents,
)
from src.document_loader import Document


# ---------------------------------------------------------------------------
# recursive_split
# ---------------------------------------------------------------------------


class TestRecursiveSplit:
    """Test recursive_split function."""

    def test_basic_split_respects_chunk_size(self):
        """Each chunk should be at most chunk_size characters."""
        text = "A" * 300
        chunks = recursive_split(text, chunk_size=100, overlap=0)
        assert len(chunks) >= 3
        assert all(len(c) <= 100 for c in chunks)

    def test_short_text_returns_single_chunk(self):
        """Text shorter than chunk_size should be returned as a single chunk."""
        text = "Short text"
        chunks = recursive_split(text, chunk_size=100, overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_empty_text_returns_empty_list(self):
        """Empty or whitespace-only text should return an empty list."""
        assert recursive_split("", chunk_size=100) == []
        assert recursive_split("   ", chunk_size=100) == []
        assert recursive_split("\n\n", chunk_size=100) == []

    def test_overlap_creates_shared_content(self):
        """Consecutive chunks should share overlapping characters."""
        text = "abcde" * 40  # 200 chars
        chunks = recursive_split(text, chunk_size=60, overlap=10)
        assert len(chunks) > 1
        # 验证相邻 chunk 之间有重叠
        for i in range(len(chunks) - 1):
            curr, nxt = chunks[i], chunks[i + 1]
            # 检查是否有至少一些共同字符
            has_overlap = any(curr[-k:] == nxt[:k] for k in range(1, min(len(curr), len(nxt)) + 1))
            assert has_overlap, f"No overlap found between chunk {i} and {i + 1}"

    def test_custom_separators(self):
        """Should split on custom separators."""
        text = "Section1::Section2::Section3"
        chunks = recursive_split(text, chunk_size=12, separators=["::"])
        assert len(chunks) == 3
        assert "Section1" in chunks[0]
        assert "Section2" in chunks[1]
        assert "Section3" in chunks[2]

    def test_chinese_separator_newlines(self):
        """Should split on Chinese-style double newlines."""
        text = "第一段落内容很长很长很长很长很长很长\n\n第二段落内容也很长很长很长很长很长很长"
        chunks = recursive_split(text, chunk_size=20, overlap=0)
        assert len(chunks) >= 2

    def test_hard_split_fallback(self):
        """When no separator works, should hard-split on character boundary."""
        text = "X" * 250
        chunks = recursive_split(text, chunk_size=100, overlap=0)
        assert all(len(c) <= 100 for c in chunks)
        assert len(chunks) >= 3


# ---------------------------------------------------------------------------
# financial_split
# ---------------------------------------------------------------------------


class TestFinancialSplit:
    """Test financial_split function."""

    def test_splits_on_chinese_article_pattern(self):
        """Should split on 第X條 patterns."""
        text = "第1條 這是第一條的內容。\n第2條 這是第二條的內容。\n第3條 這是第三條的內容。"
        chunks = financial_split(text)
        assert len(chunks) >= 3
        assert any("第1條" in c for c in chunks)
        assert any("第2條" in c for c in chunks)
        assert any("第3條" in c for c in chunks)

    def test_splits_on_english_article_pattern(self):
        """Should split on Article X patterns."""
        text = "Article 1 Introduction content.\nArticle 2 Requirements content.\nArticle 3 Conclusions."
        chunks = financial_split(text)
        assert len(chunks) >= 3
        assert any("Article 1" in c for c in chunks)
        assert any("Article 2" in c for c in chunks)

    def test_splits_on_numbered_list(self):
        """Should split on numbered list patterns like 1. ..."""
        text = "1. First point with enough content.\n2. Second point with enough content.\n3. Third point."
        chunks = financial_split(text)
        assert len(chunks) >= 2

    def test_fallback_to_paragraph_split(self):
        """Should fall back to paragraph splitting if no article pattern found."""
        text = "段落一的內容。\n\n段落二的內容。\n\n段落三的內容。"
        chunks = financial_split(text)
        assert len(chunks) >= 2

    def test_empty_text_returns_empty_list(self):
        """Empty text should return an empty list."""
        assert financial_split("") == []
        assert financial_split("   ") == []

    def test_long_segment_gets_subsplit(self):
        """Segments longer than 1000 chars should be recursively split."""
        # 创建一个超过 1000 字符的段落
        long_content = "第1條 " + "內容" * 600  # > 1000 chars
        text = long_content + "\n第2條 短內容。"
        chunks = financial_split(text)
        # 第1條的超长段落应该被进一步分割
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# split_documents
# ---------------------------------------------------------------------------


class TestSplitDocuments:
    """Test split_documents unified entry point."""

    def test_recursive_strategy(self):
        """Test recursive strategy through split_documents."""
        docs = [
            Document(
                content="A " * 100,
                metadata={"source": "test.txt", "page": None, "title": "Test", "doc_type": "text"},
            ),
            Document(
                content="B " * 100,
                metadata={"source": "test2.txt", "page": None, "title": "Test2", "doc_type": "text"},
            ),
        ]
        chunks = split_documents(docs, strategy="recursive", chunk_size=50)
        assert len(chunks) >= 4
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_preserves_metadata(self):
        """Chunks should inherit metadata from their source Document."""
        docs = [
            Document(
                content="第1條 一些足夠長的內容讓它被分割。",
                metadata={"source": "regulation.pdf", "page": 5, "title": "Regulations", "doc_type": "pdf"},
            ),
        ]
        chunks = split_documents(docs, strategy="financial")
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata.get("source") == "regulation.pdf"
            assert chunk.metadata.get("page") == 5

    def test_financial_strategy(self):
        """Test financial strategy splits on article patterns."""
        docs = [
            Document(
                content="第1條 內容一。\n第2條 內容二。\n第3條 內容三。",
                metadata={"source": "law.md", "page": None, "title": "Law", "doc_type": "markdown"},
            ),
        ]
        chunks = split_documents(docs, strategy="financial")
        assert len(chunks) >= 3

    def test_assigns_chunk_ids(self):
        """Every chunk should have a non-empty chunk_id."""
        docs = [
            Document(
                content="Some content that will be split into chunks for testing.",
                metadata={"source": "test.txt", "page": None, "title": "Test", "doc_type": "text"},
            ),
        ]
        chunks = split_documents(docs, strategy="recursive", chunk_size=30)
        assert all(c.chunk_id != "" for c in chunks)

    def test_empty_documents_returns_empty(self):
        """An empty document list should return an empty chunk list."""
        assert split_documents([], strategy="recursive") == []

    def test_empty_content_skipped(self):
        """Documents with empty or whitespace-only content should be skipped."""
        docs = [
            Document(content="", metadata={"source": "a.txt"}),
            Document(content="   ", metadata={"source": "b.txt"}),
            Document(content="Valid content here.", metadata={"source": "c.txt"}),
        ]
        chunks = split_documents(docs, strategy="recursive")
        assert len(chunks) == 1
        assert "Valid content" in chunks[0].content

    def test_unknown_strategy_raises(self):
        """Passing an unknown strategy name should raise ValueError."""
        docs = [Document(content="Test", metadata={})]
        with pytest.raises(ValueError, match="Unknown splitting strategy"):
            split_documents(docs, strategy="nonexistent")
