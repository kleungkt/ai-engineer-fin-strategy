"""Tests for src/source_tracker module."""

import pytest

from src.source_tracker import SourceTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker():
    """A fresh SourceTracker instance."""
    return SourceTracker()


@pytest.fixture
def sample_sources():
    """A list of source dicts for testing."""
    return [
        {"content": "MACD is a momentum indicator that shows moving average convergence.", "metadata": {"source": "technical_indicators.md", "title": "MACD"}, "score": 0.95},
        {"content": "Bollinger Bands measure volatility using standard deviations.", "metadata": {"source": "technical_indicators.md", "title": "Bollinger"}, "score": 0.87},
        {"content": "Risk management is crucial for trading.", "metadata": {"source": "risk_management.md", "title": "Risk"}, "score": 0.75},
    ]


# ---------------------------------------------------------------------------
# extract_citations
# ---------------------------------------------------------------------------


class TestExtractCitations:
    """Test extract_citations method."""

    def test_finds_single_citation(self, tracker):
        """Should find a single [1] citation."""
        answer = "MACD is a momentum indicator [1]."
        citations = tracker.extract_citations(answer)
        assert citations == ["1"]

    def test_finds_multiple_citations(self, tracker):
        """Should find multiple citations in order of appearance."""
        answer = "MACD [1] and Bollinger Bands [2] are popular indicators [1]."
        citations = tracker.extract_citations(answer)
        assert citations == ["1", "2"]

    def test_deduplicates_citations(self, tracker):
        """Duplicate citations should be deduplicated, preserving first occurrence."""
        answer = "See [1] and also [1] and [2]."
        citations = tracker.extract_citations(answer)
        assert citations == ["1", "2"]

    def test_no_citations(self, tracker):
        """Answer without any [N] markers should return empty list."""
        answer = "MACD is a momentum indicator."
        citations = tracker.extract_citations(answer)
        assert citations == []

    def test_empty_answer(self, tracker):
        """Empty answer should return empty list."""
        assert tracker.extract_citations("") == []

    def test_large_number_citation(self, tracker):
        """Should handle multi-digit citation numbers like [10], [12]."""
        answer = "Reference [10] and [12]."
        citations = tracker.extract_citations(answer)
        assert "10" in citations
        assert "12" in citations

    def test_ordered_by_appearance(self, tracker):
        """Citations should be in order of first appearance."""
        answer = "First [3], then [1], then [2], then [3] again."
        citations = tracker.extract_citations(answer)
        assert citations == ["3", "1", "2"]


# ---------------------------------------------------------------------------
# validate_citations
# ---------------------------------------------------------------------------


class TestValidateCitations:
    """Test validate_citations method."""

    def test_valid_citations(self, tracker, sample_sources):
        """All citations within range should be valid."""
        answer = "MACD [1] and Bollinger [2] are useful."
        assert tracker.validate_citations(answer, sample_sources) is True

    def test_invalid_citation_out_of_range(self, tracker, sample_sources):
        """Citation [5] with only 3 sources should be invalid."""
        answer = "See source [5]."
        assert tracker.validate_citations(answer, sample_sources) is False

    def test_citation_zero_is_invalid(self, tracker, sample_sources):
        """Citation [0] should be invalid (1-indexed)."""
        answer = "Source [0] is wrong."
        assert tracker.validate_citations(answer, sample_sources) is False

    def test_no_citations_is_valid(self, tracker, sample_sources):
        """Answer without citations should be considered valid."""
        answer = "MACD is a momentum indicator."
        assert tracker.validate_citations(answer, sample_sources) is True

    def test_citations_with_no_sources(self, tracker):
        """If there are citations but no sources, should be invalid."""
        answer = "See [1]."
        assert tracker.validate_citations(answer, []) is False

    def test_empty_answer_is_valid(self, tracker, sample_sources):
        """Empty answer is valid (no citations to validate)."""
        assert tracker.validate_citations("", sample_sources) is True


# ---------------------------------------------------------------------------
# format_sources
# ---------------------------------------------------------------------------


class TestFormatSources:
    """Test format_sources method."""

    def test_format_with_content(self, tracker, sample_sources):
        """Should produce a formatted string with numbered entries."""
        output = tracker.format_sources(sample_sources)
        assert "[1]" in output
        assert "[2]" in output
        assert "[3]" in output
        assert "MACD" in output
        assert "Bollinger" in output

    def test_format_includes_source_info(self, tracker):
        """Should include source file, title, and content preview."""
        chunks = [
            {"content": "Some content here", "metadata": {"source": "test.pdf", "page": 1, "title": "Test"}, "score": 0.9},
        ]
        output = tracker.format_sources(chunks)
        assert "test.pdf" in output
        assert "Test" in output
        assert "Some content here" in output

    def test_format_empty_sources(self, tracker):
        """Empty sources list should return a fallback message."""
        output = tracker.format_sources([])
        assert "無來源" in output or "无来源" in output

    def test_format_long_content_is_truncated(self, tracker):
        """Content longer than 100 chars should be truncated with '...'."""
        long_content = "A" * 200
        chunks = [
            {"content": long_content, "metadata": {"source": "a.pdf"}, "score": 0.5},
        ]
        output = tracker.format_sources(chunks)
        assert "..." in output

    def test_format_includes_score(self, tracker):
        """Score should appear in the output when available."""
        chunks = [
            {"content": "Content", "metadata": {"source": "a.pdf"}, "score": 0.876},
        ]
        output = tracker.format_sources(chunks)
        assert "0.876" in output


# ---------------------------------------------------------------------------
# get_cited_sources
# ---------------------------------------------------------------------------


class TestGetCitedSources:
    """Test get_cited_sources method."""

    def test_returns_only_cited_sources(self, tracker, sample_sources):
        """Should return only sources referenced by citations in the answer."""
        answer = "MACD [1] is useful."
        cited = tracker.get_cited_sources(answer, sample_sources)
        assert len(cited) == 1
        assert cited[0]["content"] == sample_sources[0]["content"]

    def test_multiple_citations(self, tracker, sample_sources):
        answer = "MACD [1] and Bollinger [2]."
        cited = tracker.get_cited_sources(answer, sample_sources)
        assert len(cited) == 2

    def test_no_citations_returns_empty(self, tracker, sample_sources):
        answer = "No citations here."
        cited = tracker.get_cited_sources(answer, sample_sources)
        assert cited == []

    def test_out_of_range_citation_ignored(self, tracker, sample_sources):
        """Out-of-range citations should be silently skipped."""
        answer = "MACD [1] and [99]."
        cited = tracker.get_cited_sources(answer, sample_sources)
        assert len(cited) == 1
