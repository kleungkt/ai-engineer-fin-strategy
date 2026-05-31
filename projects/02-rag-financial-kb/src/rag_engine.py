"""
Core RAG pipeline for the financial knowledge base.

Provides a complete retrieval-augmented generation pipeline with configurable
retrieval, query rewriting, and LLM generation components.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RAGResponse(BaseModel):
    """Response from the RAG pipeline.

    Attributes:
        answer: The generated answer text.
        sources: List of source documents used, each with content, source, score.
        rewritten_query: The rewritten query (if query rewriting was used).
        confidence: Confidence score for the answer (0.0–1.0).
    """

    answer: str
    sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of source documents with content, source, and score",
    )
    rewritten_query: str = ""
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------


class RAGEngine:
    """Core RAG pipeline for the financial knowledge base.

    Coordinates retrieval, query rewriting, context building, and LLM
    generation to answer questions grounded in the knowledge base.

    Pipeline steps:
        1. Optionally rewrite query for better retrieval
        2. Retrieve relevant chunks via the Retriever
        3. Build context from chunks with source markers
        4. Call LLM with context + question
        5. Return answer with sources and confidence

    Args:
        retriever: Retriever instance for document retrieval.
        llm_model: OpenAI model name for answer generation.
    """

    # 默认 RAG 提示词模板
    DEFAULT_SYSTEM_PROMPT = (
        "你是一个专业的金融助手。请仅根据提供的上下文回答用户的问题。"
        "如果上下文没有足够的信息，请说明你无法回答。"
        "请使用 [1], [2] 等标记引用来源。请使用与问题相同的语言回答。"
    )

    DEFAULT_RAG_PROMPT = """请根据以下上下文信息回答问题。

上下文:
{context}

问题: {question}

请在回答中使用 [1], [2] 等标记引用来源。"""

    # 查询改写提示词
    QUERY_REWRITE_PROMPT = """你是一个查询优化专家。请将以下用户问题改写为更适合文档检索的形式。
保留原始问题的核心意图，但使用更精确的关键词。
只输出改写后的查询，不要解释。

原始问题: {question}

改写后的查询:"""

    def __init__(
        self,
        retriever: Any,
        llm_model: str = "gpt-4o-mini",
        openai_client: Any | None = None,
    ) -> None:
        """Initialize the RAG engine.

        Args:
            retriever: Retriever instance (from retriever.py).
            llm_model: OpenAI model name for answer generation.
            openai_client: Optional pre-configured OpenAI client.

        Raises:
            ImportError: If the openai package is not installed.
        """
        self._retriever = retriever
        self._llm_model = llm_model

        if openai_client is not None:
            self._openai_client = openai_client
        else:
            try:
                from openai import OpenAI  # type: ignore[import-untyped]

                self._openai_client = OpenAI()
            except ImportError as exc:
                raise ImportError(
                    "The 'openai' package is required for RAGEngine. "
                    "Install it with: pip install openai"
                ) from exc

        logger.info("Initialized RAGEngine with model=%s", llm_model)

    def query(
        self,
        question: str,
        top_k: int = 5,
        use_rewrite: bool = True,
    ) -> RAGResponse:
        """Full RAG pipeline: rewrite → retrieve → build context → generate answer.

        Args:
            question: User's natural language question.
            top_k: Number of documents to retrieve.
            use_rewrite: Whether to use LLM to rewrite the query for better retrieval.

        Returns:
            RAGResponse with answer, sources, rewritten_query, and confidence.
        """
        # 1. 可选：使用 LLM 改写查询以提高检索质量
        rewritten_query = question
        if use_rewrite:
            try:
                rewritten_query = self.rewrite_query(question)
                logger.debug("Query rewritten: '%s' → '%s'", question, rewritten_query)
            except Exception as e:
                logger.warning("Query rewriting failed, using original: %s", e)
                rewritten_query = question

        # 2. 检索相关文档
        retrieved_chunks = self._retriever.similarity_search(
            rewritten_query, top_k=top_k
        )

        # 3. 构建上下文
        context = self.build_context(retrieved_chunks)

        # 4. 使用 LLM 生成回答
        answer = self._generate_answer(question, context)

        # 5. 构建 sources 列表
        sources = [
            {
                "content": chunk.get("content", ""),
                "source": chunk.get("metadata", {}).get("source", ""),
                "score": 1.0 - chunk.get("distance", 0.0),  # 转为相似度分数
            }
            for chunk in retrieved_chunks
        ]

        # 6. 计算置信度（基于检索结果的质量）
        confidence = self._estimate_confidence(retrieved_chunks, answer)

        return RAGResponse(
            answer=answer,
            sources=sources,
            rewritten_query=rewritten_query if use_rewrite else "",
            confidence=confidence,
        )

    def rewrite_query(self, question: str) -> str:
        """Use LLM to rewrite query for better retrieval.

        Reformulates the question to be more precise and use better
        keywords for document retrieval.

        Args:
            question: Original user question.

        Returns:
            Rewritten query string.
        """
        prompt = self.QUERY_REWRITE_PROMPT.format(question=question)

        try:
            response = self._openai_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": "你是一个查询优化专家。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            rewritten = response.choices[0].message.content
            return rewritten.strip() if rewritten else question
        except Exception as e:
            logger.warning("Query rewriting failed: %s", e)
            return question

    def build_context(self, chunks: list[dict[str, Any]]) -> str:
        """Format chunks into context string with source markers.

        Each chunk is prefixed with a numbered citation marker [1], [2], etc.

        Args:
            chunks: List of chunk dicts from retrieval (content, metadata, distance).

        Returns:
            Formatted context string with source markers.
        """
        if not chunks:
            return "没有找到相关上下文。"

        context_parts = []
        for idx, chunk in enumerate(chunks, start=1):
            content = chunk.get("content", "")
            source = chunk.get("metadata", {}).get("source", "")
            page = chunk.get("metadata", {}).get("page", "")
            title = chunk.get("metadata", {}).get("title", "")

            # 构建来源信息
            source_info = f"来源: {source}"
            if page:
                source_info += f", 页码: {page}"
            if title:
                source_info += f", 标题: {title}"

            context_parts.append(f"[{idx}] ({source_info})\n{content}")

        return "\n\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate an answer using the LLM with context and question.

        Args:
            question: Original user question.
            context: Formatted context string from retrieved documents.

        Returns:
            Generated answer string.
        """
        prompt = self.DEFAULT_RAG_PROMPT.format(
            question=question,
            context=context,
        )

        try:
            response = self._openai_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1000,
            )
            return response.choices[0].message.content or "无法生成回答。"
        except Exception as exc:
            logger.error("Failed to generate answer: %s", exc)
            return f"生成回答时出错: {exc}"

    def _estimate_confidence(
        self,
        chunks: list[dict[str, Any]],
        answer: str,
    ) -> float:
        """Estimate confidence score for the answer.

        Based on:
        - Average retrieval distance (lower = better)
        - Number of relevant chunks found
        - Whether the answer references sources

        Args:
            chunks: Retrieved chunks.
            answer: Generated answer.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not chunks:
            return 0.0

        # 因子 1: 检索质量（平均距离越低越好）
        distances = [c.get("distance", 1.0) for c in chunks]
        avg_distance = sum(distances) / len(distances)
        retrieval_quality = max(0.0, 1.0 - avg_distance)

        # 因子 2: 检索数量（有足够多的相关文档）
        quantity_factor = min(1.0, len(chunks) / 3.0)  # 至少3个文档给满分

        # 因子 3: 回答中是否包含引用标记
        citation_count = answer.count("[") + answer.count("]")
        citation_factor = min(1.0, citation_count / 4.0) if citation_count > 0 else 0.3

        # 加权组合
        confidence = (
            0.5 * retrieval_quality
            + 0.25 * quantity_factor
            + 0.25 * citation_factor
        )

        return round(min(1.0, max(0.0, confidence)), 2)
