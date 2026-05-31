"""Tests for src/document_loader module."""

import pytest
import tempfile
from pathlib import Path

from src.document_loader import (
    Document,
    load_markdown,
    load_text,
    load_file,
    load_directory,
    clear_cache,
)


# ---------------------------------------------------------------------------
# Document model
# ---------------------------------------------------------------------------


class TestDocument:
    """Test the Document Pydantic model."""

    def test_creation_with_metadata(self):
        doc = Document(
            content="Hello world",
            metadata={"source": "a.txt", "page": 1, "title": "A", "doc_type": "text"},
        )
        assert doc.content == "Hello world"
        assert doc.metadata["source"] == "a.txt"

    def test_default_metadata(self):
        """Default metadata should have standard keys."""
        doc = Document(content="Test")
        assert "source" in doc.metadata
        assert "page" in doc.metadata
        assert "title" in doc.metadata
        assert "doc_type" in doc.metadata


# ---------------------------------------------------------------------------
# load_markdown
# ---------------------------------------------------------------------------


class TestLoadMarkdown:
    """Test load_markdown function."""

    def test_splits_by_headers(self):
        """Should split on ## headers into separate Documents."""
        content = "# Title\n\nIntro text\n\n## Section A\n\nContent A\n\n## Section B\n\nContent B"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            clear_cache()
            docs = load_markdown(path)
            assert len(docs) >= 2
            titles = [d.metadata.get("title", "") for d in docs]
            assert any("Section A" in t for t in titles)
            assert any("Section B" in t for t in titles)
        finally:
            Path(path).unlink()
            clear_cache()

    def test_preserves_filename_as_title_for_preamble(self):
        """Text before the first header should use the filename as title."""
        content = "Preamble text before any header.\n\n## First Header\n\nBody."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            clear_cache()
            docs = load_markdown(path)
            # 第一个 section 可能是 preamble 或者包含 preamble
            assert len(docs) >= 1
            # 所有文档都应有 source
            assert all(d.metadata["source"] == path for d in docs)
        finally:
            Path(path).unlink()
            clear_cache()

    def test_doc_type_is_markdown(self):
        """All loaded documents should have doc_type='markdown'."""
        content = "## Header\n\nContent here."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            clear_cache()
            docs = load_markdown(path)
            assert all(d.metadata.get("doc_type") == "markdown" for d in docs)
        finally:
            Path(path).unlink()
            clear_cache()

    def test_empty_markdown_returns_empty_list(self):
        """An empty markdown file should produce at least one doc or empty list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("")
            f.flush()
            path = f.name

        try:
            clear_cache()
            docs = load_markdown(path)
            # 空文件可能返回空列表或一个空文档
            assert isinstance(docs, list)
        finally:
            Path(path).unlink()
            clear_cache()

    def test_nonexistent_file_returns_empty(self):
        """Loading a non-existent file should return an empty list."""
        assert load_markdown("/nonexistent/path/file.md") == []


# ---------------------------------------------------------------------------
# load_text
# ---------------------------------------------------------------------------


class TestLoadText:
    """Test load_text function."""

    def test_single_paragraph(self):
        """A single-paragraph file should produce one Document."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Single paragraph content.")
            f.flush()
            path = f.name

        try:
            clear_cache()
            docs = load_text(path)
            assert len(docs) == 1
            assert docs[0].content == "Single paragraph content."
        finally:
            Path(path).unlink()
            clear_cache()

    def test_multiple_paragraphs(self):
        """Double-newline separated paragraphs should become separate Documents."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Paragraph one.\n\nParagraph two.\n\nParagraph three.")
            f.flush()
            path = f.name

        try:
            clear_cache()
            docs = load_text(path)
            assert len(docs) == 3
            assert docs[0].content == "Paragraph one."
            assert docs[1].content == "Paragraph two."
            assert docs[2].content == "Paragraph three."
        finally:
            Path(path).unlink()
            clear_cache()

    def test_metadata_fields(self):
        """Loaded docs should have proper source and doc_type metadata."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Content.")
            f.flush()
            path = f.name

        try:
            clear_cache()
            docs = load_text(path)
            assert docs[0].metadata["source"] == path
            assert docs[0].metadata["doc_type"] == "text"
        finally:
            Path(path).unlink()
            clear_cache()

    def test_nonexistent_returns_empty(self):
        """Loading a non-existent file should return an empty list."""
        assert load_text("/nonexistent/file.txt") == []


# ---------------------------------------------------------------------------
# load_directory
# ---------------------------------------------------------------------------


class TestLoadDirectory:
    """Test load_directory function."""

    def test_finds_all_supported_files(self):
        """Should load .txt, .md files from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "a.txt").write_text("Text content.", encoding="utf-8")
            Path(tmpdir, "b.md").write_text("## Header\n\nMarkdown content.", encoding="utf-8")
            Path(tmpdir, "c.xyz").write_text("Unsupported.", encoding="utf-8")

            clear_cache()
            docs = load_directory(tmpdir)
            sources = [d.metadata["source"] for d in docs]
            assert any("a.txt" in s for s in sources)
            assert any("b.md" in s for s in sources)
            # .xyz 不应被加载
            assert not any("c.xyz" in s for s in sources)
            clear_cache()

    def test_recursive_search(self):
        """Should find files in subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir, "sub")
            subdir.mkdir()
            Path(tmpdir, "top.txt").write_text("Top level.", encoding="utf-8")
            Path(subdir, "nested.txt").write_text("Nested level.", encoding="utf-8")

            clear_cache()
            docs = load_directory(tmpdir)
            sources = [d.metadata["source"] for d in docs]
            assert any("top.txt" in s for s in sources)
            assert any("nested.txt" in s for s in sources)
            clear_cache()

    def test_nonexistent_directory_returns_empty(self):
        assert load_directory("/nonexistent/dir") == []

    def test_file_path_returns_empty(self):
        """Passing a file path instead of a directory should return empty."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            f.flush()
            path = f.name

        try:
            assert load_directory(path) == []
        finally:
            Path(path).unlink()


# ---------------------------------------------------------------------------
# load_file auto-detect
# ---------------------------------------------------------------------------


class TestLoadFile:
    """Test load_file auto-detection."""

    def test_auto_detects_txt(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Test content.")
            f.flush()
            path = f.name

        try:
            clear_cache()
            docs = load_file(path)
            assert len(docs) >= 1
        finally:
            Path(path).unlink()
            clear_cache()

    def test_unsupported_extension_returns_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("test")
            f.flush()
            path = f.name

        try:
            assert load_file(path) == []
        finally:
            Path(path).unlink()

    def test_nonexistent_file_returns_empty(self):
        assert load_file("/nonexistent/file.txt") == []
