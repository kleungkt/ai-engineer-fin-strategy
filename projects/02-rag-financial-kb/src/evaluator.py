"""
RAG evaluation module for the financial knowledge base.

Provides evaluation utilities for assessing retrieval quality (Precision@K,
Recall@K, MRR) and answer quality (faithfulness, relevancy).

Uses simple heuristics: keyword overlap for faithfulness, semantic similarity
for relevancy. Can be extended with more sophisticated metrics.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Tokenize text into a set of lowercase tokens for keyword matching.

    Handles both Chinese characters and English words.

    Args:
        text: Input text.

    Returns:
        Set of lowercase tokens.
    """
    # 匹配中文字符、英文单词和数字
    tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+|\d+", text.lower())
    # 对于中文，也把连续的字符拆成单字
    expanded: set[str] = set()
    for token in tokens:
        if re.match(r"[\u4e00-\u9fff]+", token):
            # 中文：添加整个词和每个单字
            expanded.add(token)
            for char in token:
                expanded.add(char)
        else:
            expanded.add(token)
    return expanded


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Compute keyword overlap ratio between two texts.

    Uses Jaccard similarity of token sets.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Overlap ratio between 0.0 and 1.0.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    return len(intersection) / len(union) if union else 0.0


def _simple_embed_similarity(text_a: str, text_b: str) -> float:
    """Compute simple character n-gram similarity between two texts.

    Uses character bigram overlap as a proxy for semantic similarity.
    No external model needed.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Similarity score between 0.0 and 1.0.
    """

    def _bigrams(text: str) -> dict[str, int]:
        bg: dict[str, int] = {}
        for i in range(len(text) - 1):
            pair = text[i : i + 2]
            bg[pair] = bg.get(pair, 0) + 1
        return bg

    bg_a = _bigrams(text_a.lower())
    bg_b = _bigrams(text_b.lower())

    if not bg_a or not bg_b:
        return 0.0

    # Cosine similarity of bigram vectors
    all_keys = set(bg_a.keys()) | set(bg_b.keys())
    dot = sum(bg_a.get(k, 0) * bg_b.get(k, 0) for k in all_keys)
    norm_a = sum(v * v for v in bg_a.values()) ** 0.5
    norm_b = sum(v * v for v in bg_b.values()) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class RAGEvaluator:
    """Evaluate RAG pipeline quality.

    Provides methods to evaluate:
    - Retrieval quality: Precision@K, Recall@K, Mean Reciprocal Rank (MRR)
    - Answer quality: Faithfulness (grounded in context), Relevancy (addresses question)

    Usage::

        evaluator = RAGEvaluator()
        retrieval_metrics = evaluator.evaluate_retrieval(queries, expected_docs, retriever)
        answer_metrics = evaluator.evaluate_answer(question, answer, context)
        full_report = evaluator.run_evaluation(test_cases)
    """

    def evaluate_retrieval(
        self,
        queries: list[str],
        expected_docs: list[list[str]],
        retriever: Any,
        top_k: int = 5,
    ) -> dict[str, float]:
        """Compute retrieval quality metrics: Precision@K, Recall@K, MRR.

        Args:
            queries: List of query strings.
            expected_docs: List of lists of expected document content snippets.
                Each inner list contains the relevant document contents for
                the corresponding query.
            retriever: Retriever instance with similarity_search method.
            top_k: Number of results to retrieve for each query.

        Returns:
            Dict with keys: precision@k, recall@k, mrr.
        """
        if len(queries) != len(expected_docs):
            raise ValueError(
                f"queries ({len(queries)}) and expected_docs ({len(expected_docs)}) "
                f"must have the same length"
            )

        precisions: list[float] = []
        recalls: list[float] = []
        reciprocal_ranks: list[float] = []

        for query, expected in zip(queries, expected_docs):
            # 检索结果
            results = retriever.similarity_search(query, top_k=top_k)
            retrieved_contents = [r.get("content", "") for r in results]

            # 计算 Precision@K: 检索结果中有多少是相关的
            relevant_count = 0
            first_relevant_rank = 0

            for rank, content in enumerate(retrieved_contents, start=1):
                is_relevant = False
                for expected_doc in expected:
                    if _keyword_overlap(content, expected_doc) > 0.1:
                        is_relevant = True
                        break

                if is_relevant:
                    relevant_count += 1
                    if first_relevant_rank == 0:
                        first_relevant_rank = rank

            precision = relevant_count / top_k if top_k > 0 else 0.0
            recall = relevant_count / len(expected) if expected else 0.0
            mrr = 1.0 / first_relevant_rank if first_relevant_rank > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            reciprocal_ranks.append(mrr)

        # 计算平均值
        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        avg_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

        metrics = {
            "precision@k": round(avg_precision, 4),
            "recall@k": round(avg_recall, 4),
            "mrr": round(avg_mrr, 4),
            "num_queries": len(queries),
        }

        logger.info(
            "Retrieval evaluation: P@%d=%.3f, R@%d=%.3f, MRR=%.3f",
            top_k,
            avg_precision,
            top_k,
            avg_recall,
            avg_mrr,
        )

        return metrics

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> dict[str, float]:
        """Evaluate answer quality: faithfulness and relevancy.

        Faithfulness: How well the answer is grounded in the provided context.
            Measured by keyword overlap between answer and context.
        Relevancy: How well the answer addresses the question.
            Measured by semantic similarity between question and answer.

        Args:
            question: The original question.
            answer: The generated answer.
            context: The context used for generation.

        Returns:
            Dict with keys: faithfulness, relevancy, overall.
        """
        # 信仰度：回答中有多少内容基于上下文
        # 使用关键词重叠度衡量
        faithfulness = _keyword_overlap(answer, context)

        # 相关性：回答是否针对问题
        # 使用字符 n-gram 相似度衡量
        relevancy = _simple_embed_similarity(question, answer)

        # 综合分数
        overall = 0.5 * faithfulness + 0.5 * relevancy

        metrics = {
            "faithfulness": round(faithfulness, 4),
            "relevancy": round(relevancy, 4),
            "overall": round(overall, 4),
        }

        logger.info(
            "Answer evaluation: faithfulness=%.3f, relevancy=%.3f, overall=%.3f",
            faithfulness,
            relevancy,
            overall,
        )

        return metrics

    def run_evaluation(
        self,
        test_cases: list[dict[str, Any]],
        retriever: Any | None = None,
        rag_engine: Any | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Run full evaluation suite on a set of test cases.

        Test case format::

            {
                "query": "What is compound interest?",
                "expected_answer_keywords": ["compound", "interest", "principal"],
                "expected_sources": ["source_doc_content_snippet_1", ...],
            }

        Args:
            test_cases: List of test case dicts.
            retriever: Retriever instance for retrieval evaluation.
            rag_engine: RAGEngine instance for end-to-end evaluation.
            top_k: Number of results to retrieve.

        Returns:
            Dict with aggregated metrics and per-case details.
        """
        results: dict[str, Any] = {
            "total_cases": len(test_cases),
            "retrieval_metrics": {},
            "answer_metrics": {},
            "per_case_results": [],
        }

        retrieval_precisions: list[float] = []
        retrieval_recalls: list[float] = []
        retrieval_mrrs: list[float] = []
        answer_faithfulness: list[float] = []
        answer_relevancy: list[float] = []

        for i, case in enumerate(test_cases):
            query = case.get("query", "")
            expected_keywords = case.get("expected_answer_keywords", [])
            expected_sources = case.get("expected_sources", [])

            case_result: dict[str, Any] = {
                "query": query,
                "case_index": i,
            }

            # 检索评估
            if retriever is not None:
                retrieved = retriever.similarity_search(query, top_k=top_k)
                retrieved_contents = [r.get("content", "") for r in retrieved]

                # Precision@K
                relevant_count = 0
                first_relevant = 0
                for rank, content in enumerate(retrieved_contents, start=1):
                    for expected_src in expected_sources:
                        if _keyword_overlap(content, expected_src) > 0.1:
                            relevant_count += 1
                            if first_relevant == 0:
                                first_relevant = rank
                            break

                precision = relevant_count / top_k if top_k > 0 else 0.0
                recall = relevant_count / len(expected_sources) if expected_sources else 0.0
                mrr = 1.0 / first_relevant if first_relevant > 0 else 0.0

                retrieval_precisions.append(precision)
                retrieval_recalls.append(recall)
                retrieval_mrrs.append(mrr)

                case_result["retrieval"] = {
                    "precision": precision,
                    "recall": recall,
                    "mrr": mrr,
                    "num_retrieved": len(retrieved_contents),
                }

            # 端到端评估（使用 RAG engine）
            if rag_engine is not None:
                response = rag_engine.query(query, top_k=top_k, use_rewrite=False)
                answer = response.answer

                # 构建上下文
                context = " ".join(s.get("content", "") for s in response.sources)

                # 信仰度
                faithfulness = _keyword_overlap(answer, context)

                # 相关性（使用关键词匹配）
                if expected_keywords:
                    answer_tokens = _tokenize(answer)
                    keyword_matches = sum(
                        1 for kw in expected_keywords if kw.lower() in answer
                    )
                    relevancy = keyword_matches / len(expected_keywords)
                else:
                    relevancy = _simple_embed_similarity(query, answer)

                answer_faithfulness.append(faithfulness)
                answer_relevancy.append(relevancy)

                case_result["answer"] = {
                    "faithfulness": faithfulness,
                    "relevancy": relevancy,
                    "answer_preview": answer[:200],
                }

            results["per_case_results"].append(case_result)

        # 汇总指标
        if retrieval_precisions:
            results["retrieval_metrics"] = {
                "precision@k": round(sum(retrieval_precisions) / len(retrieval_precisions), 4),
                "recall@k": round(sum(retrieval_recalls) / len(retrieval_recalls), 4),
                "mrr": round(sum(retrieval_mrrs) / len(retrieval_mrrs), 4),
                "num_queries": len(retrieval_precisions),
            }

        if answer_faithfulness:
            results["answer_metrics"] = {
                "faithfulness": round(sum(answer_faithfulness) / len(answer_faithfulness), 4),
                "relevancy": round(sum(answer_relevancy) / len(answer_relevancy), 4),
                "overall": round(
                    (
                        sum(answer_faithfulness) / len(answer_faithfulness)
                        + sum(answer_relevancy) / len(answer_relevancy)
                    )
                    / 2,
                    4,
                ),
                "num_evaluated": len(answer_faithfulness),
            }

        logger.info(
            "Evaluation complete: %d cases, retrieval=%s, answer=%s",
            len(test_cases),
            results.get("retrieval_metrics", {}),
            results.get("answer_metrics", {}),
        )

        return results
