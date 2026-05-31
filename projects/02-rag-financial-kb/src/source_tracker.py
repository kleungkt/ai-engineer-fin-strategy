"""
Source tracking and citation management for RAG financial knowledge base.

Provides utilities to extract, format, and validate source attributions
in RAG-generated answers.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class SourceTracker:
    """Track and format source attributions in RAG answers.

    Handles citation extraction (e.g., [1], [2]), source formatting
    for display, and validation that citations reference valid sources.

    Usage::

        tracker = SourceTracker()
        citations = tracker.extract_citations(answer)
        formatted = tracker.format_sources(chunks)
        is_valid = tracker.validate_citations(answer, chunks)
    """

    # 匹配 [1], [2], [3] 等引用标记的正则表达式
    _CITATION_PATTERN = re.compile(r"\[(\d+)\]")

    def extract_citations(self, answer: str) -> list[str]:
        """Extract [1], [2] style citations from the answer text.

        Args:
            answer: The RAG-generated answer text.

        Returns:
            List of unique citation identifiers (e.g., ['1', '2', '3']).
            Sorted in order of first appearance.
        """
        if not answer:
            return []

        matches = self._CITATION_PATTERN.findall(answer)

        # 去重并保持出现顺序
        seen: set[str] = set()
        citations: list[str] = []
        for match in matches:
            if match not in seen:
                seen.add(match)
                citations.append(match)

        logger.debug("Extracted %d citations from answer", len(citations))
        return citations

    def format_sources(self, chunks: list[dict[str, Any]]) -> str:
        """Format source list for display.

        Creates a numbered list of sources with content preview,
        source file, page number, and title.

        Args:
            chunks: List of chunk dicts with content and metadata.

        Returns:
            Formatted string with numbered source list.
        """
        if not chunks:
            return "无来源信息。"

        lines = ["来源列表:", "=" * 50]

        for idx, chunk in enumerate(chunks, start=1):
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "未知来源")
            page = metadata.get("page", "")
            title = metadata.get("title", "")
            score = chunk.get("score", chunk.get("distance", ""))

            # 内容预览（截取前100字符）
            preview = content[:100] + "..." if len(content) > 100 else content

            lines.append(f"\n[{idx}]")
            lines.append(f"  来源: {source}")
            if page:
                lines.append(f"  页码: {page}")
            if title:
                lines.append(f"  标题: {title}")
            if score != "":
                if isinstance(score, float):
                    lines.append(f"  相关度: {score:.3f}")
                else:
                    lines.append(f"  相关度: {score}")
            lines.append(f"  内容: {preview}")

        return "\n".join(lines)

    def validate_citations(
        self,
        answer: str,
        sources: list[dict[str, Any]],
    ) -> bool:
        """Check if all citations in the answer reference valid sources.

        A citation [N] is valid if N is between 1 and len(sources) inclusive.

        Args:
            answer: The RAG-generated answer text.
            sources: List of source dicts.

        Returns:
            True if all citations are valid, False otherwise.
        """
        citations = self.extract_citations(answer)

        if not citations:
            # 没有引用标记时，视为有效（不需要引用的情况）
            return True

        max_source_idx = len(sources)
        if max_source_idx == 0:
            # 有引用但没有来源，视为无效
            logger.warning("Citations found but no sources provided")
            return False

        for citation in citations:
            try:
                idx = int(citation)
                if idx < 1 or idx > max_source_idx:
                    logger.warning(
                        "Invalid citation [%d]: out of range (1-%d)",
                        idx,
                        max_source_idx,
                    )
                    return False
            except ValueError:
                logger.warning("Invalid citation format: [%s]", citation)
                return False

        logger.debug("All %d citations validated successfully", len(citations))
        return True

    def get_cited_sources(
        self,
        answer: str,
        sources: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract only the sources that are actually cited in the answer.

        Args:
            answer: The RAG-generated answer text.
            sources: List of all available source dicts.

        Returns:
            List of source dicts that are referenced in the answer.
        """
        citations = self.extract_citations(answer)
        cited_sources: list[dict[str, Any]] = []

        for citation in citations:
            try:
                idx = int(citation) - 1  # 转为 0-based index
                if 0 <= idx < len(sources):
                    cited_sources.append(sources[idx])
            except (ValueError, IndexError):
                continue

        return cited_sources
