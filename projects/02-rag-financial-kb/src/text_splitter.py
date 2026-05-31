"""
Text splitter module for RAG financial knowledge base.

Provides multiple chunking strategies:
- recursive_split: Recursive character text splitting with Chinese support
- semantic_split: Semantic splitting at sentence boundaries where similarity drops
- financial_split: Financial-document-aware splitting (articles, clauses)
- split_documents: Unified entry point that applies strategy to Documents
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class TextChunk(BaseModel):
    """A text chunk produced by any splitting strategy.

    Attributes:
        content: The text content of this chunk.
        metadata: Inherited from the source Document (source, page, title, doc_type).
        chunk_id: Unique identifier for this chunk.
        start_idx: Character start index within the original document.
        end_idx: Character end index within the original document.
    """

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_id: str = ""
    start_idx: int = 0
    end_idx: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 句子结尾标点（中英文）
_SENTENCE_END = re.compile(r"(?<=[.!?。！？；\n])\s*")

# 金融文件条文标题模式
_ARTICLE_RE = re.compile(
    r"(?:^|\n)\s*"
    r"(?:"
    r"第[一二三四五六七八九十百千零\d]+[條章節款項]"  # 第X條 / 第X章 / 第X節
    r"|Article\s+\d+"  # Article X
    r"|Section\s+\d+"  # Section X
    r"|\d+\.\s+"  # 1. ...
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def _generate_chunk_id(content: str, source: str, idx: int) -> str:
    """Generate a deterministic chunk ID from content hash and index."""
    digest = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
    return f"{source}::chunk_{idx}_{digest}"


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using Chinese and English punctuation boundaries."""
    parts: list[str] = []
    for segment in _SENTENCE_END.split(text):
        s = segment.strip()
        if s:
            parts.append(s)
    return parts


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors (simple implementation)."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _simple_embed(text: str, dim: int = 64) -> list[float]:
    """Create a simple TF-based vector for similarity comparison.

    使用字符 n-gram 的词频作为简易向量表示，用于语义分割时的
    相似度计算。不需要外部 embedding 模型。
    """
    # Character bigrams as features
    bigrams: dict[str, int] = {}
    for i in range(len(text) - 1):
        bg = text[i : i + 2]
        bigrams[bg] = bigrams.get(bg, 0) + 1

    # Hash bigrams into fixed-dim vector
    vector = [0.0] * dim
    for bg, count in bigrams.items():
        idx = hash(bg) % dim
        vector[idx] += float(count)

    # Normalize
    norm = sum(v * v for v in vector) ** 0.5
    if norm > 0:
        vector = [v / norm for v in vector]
    return vector


# ---------------------------------------------------------------------------
# recursive_split
# ---------------------------------------------------------------------------

# 默认分隔符列表（中英文混合）
_DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", "。", "，", ". ", " ", ""]


def recursive_split(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separators: list[str] | None = None,
) -> list[str]:
    """Recursively split *text* using a separator hierarchy.

    Supports both Chinese and English separators. Tries each separator
    in order (most meaningful first) and falls back to harder splits.

    Args:
        text: Input text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.
        separators: Ordered list of separators. Defaults to
            ``['\\n\\n', '\\n', '。', '，', '. ', ' ', '']``.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    if separators is None:
        separators = list(_DEFAULT_SEPARATORS)

    # Base case: text is short enough
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Try each separator
    chunks: list[str] = []
    for i, sep in enumerate(separators):
        if sep == "":
            # Hard split on character boundary with overlap
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start += chunk_size - overlap
                if start >= len(text):
                    break
            return chunks if chunks else [text.strip()]

        if sep in text:
            parts = text.split(sep)
            current: list[str] = []
            current_len = 0

            for part in parts:
                candidate_len = current_len + (len(sep) if current else 0) + len(part)
                if candidate_len <= chunk_size or not current:
                    current.append(part)
                    current_len = candidate_len
                else:
                    # 发射已累积的 chunk
                    merged = sep.join(current).strip()
                    if merged:
                        if len(merged) > chunk_size:
                            # 递归使用剩余分隔符继续分割
                            sub_chunks = recursive_split(
                                merged, chunk_size, overlap, separators[i + 1 :]
                            )
                            chunks_list = list(sub_chunks)
                        else:
                            chunks_list = [merged]
                    else:
                        chunks_list = []

                    # 应用 overlap：从上一个 chunk 的末尾取 overlap 字符
                    if chunks_list and overlap > 0 and len(chunks_list[-1]) > overlap:
                        last = chunks_list[-1]
                        overlap_text = last[-overlap:]
                        current = [overlap_text + sep + part]
                        current_len = len(current[0])
                    else:
                        current = [part]
                        current_len = len(part)

                    # 将已发射的 chunks 累积到主列表
                    chunks.extend(chunks_list)

            # Flush remaining
            if current:
                merged = sep.join(current).strip()
                if merged:
                    if len(merged) > chunk_size:
                        sub_chunks = recursive_split(
                            merged, chunk_size, overlap, separators[i + 1 :]
                        )
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(merged)

            return chunks if chunks else [text.strip()]

    # Fallback: no separator matched — hard character split
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
        if start >= len(text):
            break
    return chunks if chunks else [text.strip()]


# ---------------------------------------------------------------------------
# semantic_split
# ---------------------------------------------------------------------------


def semantic_split(
    text: str,
    threshold: float = 0.5,
    min_chunk_size: int = 50,
    max_chunk_size: int = 1000,
) -> list[str]:
    """Split text at sentence boundaries where semantic similarity drops.

    Computes pairwise similarity between adjacent sentences using character
    n-gram vectors. When the similarity between consecutive sentences drops
    below the threshold, a split point is inserted.

    Args:
        text: Input text to split.
        threshold: Similarity threshold below which to split (0.0–1.0).
        min_chunk_size: Minimum characters per chunk.
        max_chunk_size: Maximum characters per chunk.

    Returns:
        List of text chunks split at semantically coherent boundaries.
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    # 计算每个句子的简易向量
    sentence_vectors = [_simple_embed(s) for s in sentences]

    # 计算相邻句子间的相似度，找出低相似度的分割点
    split_points: list[int] = [0]  # 总是从第一个句子开始
    for i in range(len(sentences) - 1):
        sim = _cosine_similarity(sentence_vectors[i], sentence_vectors[i + 1])
        if sim < threshold:
            split_points.append(i + 1)
    split_points.append(len(sentences))

    # 根据分割点合并句子为 chunks
    chunks: list[str] = []
    for j in range(len(split_points) - 1):
        start = split_points[j]
        end = split_points[j + 1]
        segment = " ".join(sentences[start:end]).strip()

        if not segment:
            continue

        # 如果 segment 太长，用 recursive_split 再切
        if len(segment) > max_chunk_size:
            sub_chunks = recursive_split(segment, chunk_size=max_chunk_size, overlap=0)
            chunks.extend(sub_chunks)
        # 如果 segment 太短，尝试与前一个 chunk 合并
        elif len(segment) < min_chunk_size and chunks:
            merged = chunks[-1] + " " + segment
            if len(merged) <= max_chunk_size:
                chunks[-1] = merged
            else:
                chunks.append(segment)
        else:
            chunks.append(segment)

    return chunks


# ---------------------------------------------------------------------------
# financial_split
# ---------------------------------------------------------------------------


def financial_split(text: str) -> list[str]:
    """Split financial documents respecting article/clause numbering.

    Splits on patterns like:
    - 第X條, 第X章, 第X節 (Traditional Chinese legal numbering)
    - Article X, Section X (English legal numbering)
    - 1., 2., ... (Numbered lists)

    Falls back to paragraph splitting if no article patterns are found.

    Args:
        text: Input financial document text.

    Returns:
        List of text chunks split at article/clause boundaries.
    """
    if not text or not text.strip():
        return []

    # 使用前瞻断言在条文标题处分割
    article_pattern = re.compile(
        r"(?<=\n)(?=第[一二三四五六七八九十百千零\d]+[條章節款項])|"
        r"(?<=\n)(?=Article\s+\d+)|"
        r"(?<=\n)(?=Section\s+\d+)|"
        r"(?<=\n)(?=\d+\.\s+)|"
        r"(?=^第[一二三四五六七八九十百千零\d]+[條章節款項])|"
        r"(?=^Article\s+\d+)|"
        r"(?=^Section\s+\d+)|"
        r"(?=^\d+\.\s+)",
        re.IGNORECASE | re.MULTILINE,
    )

    # Find all split points
    splits = [0] + [m.start() for m in article_pattern.finditer(text)] + [len(text)]

    # Build raw segments
    segments: list[str] = []
    for i in range(len(splits) - 1):
        seg = text[splits[i] : splits[i + 1]].strip()
        if seg:
            segments.append(seg)

    # 如果没找到条文分割点，回退到段落分割
    if len(segments) <= 1:
        segments = [s.strip() for s in re.split(r"\n\s*\n", text) if s.strip()]

    # 对过长的段落用 recursive_split 进一步分割
    final_chunks: list[str] = []
    for seg in segments:
        if len(seg) > 1000:
            sub = recursive_split(seg, chunk_size=500, overlap=50)
            final_chunks.extend(sub)
        else:
            final_chunks.append(seg)

    return final_chunks


# ---------------------------------------------------------------------------
# split_documents (unified entry point)
# ---------------------------------------------------------------------------

_STRATEGY_MAP: dict[str, Any] = {
    "recursive": lambda text, **kw: recursive_split(
        text,
        chunk_size=kw.get("chunk_size", 500),
        overlap=kw.get("overlap", 50),
    ),
    "semantic": lambda text, **kw: semantic_split(
        text,
        threshold=kw.get("threshold", 0.5),
    ),
    "financial": lambda text, **kw: financial_split(text),
}


def split_documents(
    docs: list[Any],
    strategy: str = "recursive",
    **kwargs: Any,
) -> list[TextChunk]:
    """Apply the chosen splitting strategy to a list of Documents.

    Preserves metadata from each source Document and assigns chunk_ids,
    start_idx, and end_idx to each produced TextChunk.

    Args:
        docs: List of Document instances (from document_loader).
        strategy: One of 'recursive', 'semantic', or 'financial'.
        **kwargs: Extra keyword arguments forwarded to the chosen strategy.

    Returns:
        List of TextChunk instances.
    """
    splitter = _STRATEGY_MAP.get(strategy)
    if splitter is None:
        raise ValueError(
            f"Unknown splitting strategy '{strategy}'. "
            f"Choose from: {list(_STRATEGY_MAP.keys())}"
        )

    all_chunks: list[TextChunk] = []
    global_idx = 0

    for doc in docs:
        text = doc.content
        if not text or not text.strip():
            continue

        # 获取文档级别的源信息
        source = doc.metadata.get("source", "")
        text_pieces = splitter(text, **kwargs)

        # 在原文中追踪每个 chunk 的位置
        search_start = 0
        for piece in text_pieces:
            # 在原文中定位 chunk 的起止位置
            pos = text.find(piece, search_start)
            if pos == -1:
                # 如果找不到精确匹配，使用近似位置
                start_idx = search_start
                end_idx = search_start + len(piece)
            else:
                start_idx = pos
                end_idx = pos + len(piece)
                search_start = pos + len(piece)

            chunk_id = _generate_chunk_id(piece, source, global_idx)

            all_chunks.append(
                TextChunk(
                    content=piece,
                    metadata=dict(doc.metadata),  # 继承源文档 metadata
                    chunk_id=chunk_id,
                    start_idx=start_idx,
                    end_idx=end_idx,
                )
            )
            global_idx += 1

    logger.info(
        "Split %d documents into %d chunks using '%s' strategy",
        len(docs),
        len(all_chunks),
        strategy,
    )
    return all_chunks
