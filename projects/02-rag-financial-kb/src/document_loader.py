"""
Document loader module for RAG financial knowledge base.

Provides functions to load documents from various file formats (PDF via PyMuPDF,
Markdown, plain text) into a unified Document model for downstream processing.
Each loader extracts metadata including filename, page number, and section title.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------


class Document(BaseModel):
    """A single document section produced by a loader.

    Attributes:
        content: The text content of this document section.
        metadata: Dictionary with keys: source, page, title, doc_type.
    """

    content: str
    metadata: dict[str, Any] = Field(
        default_factory=lambda: {
            "source": "",
            "page": None,
            "title": "",
            "doc_type": "",
        }
    )


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

_CACHE: dict[str, list[Document]] = {}


def _cache_key(path: Path) -> str:
    """Return a stable cache key derived from the file path and size."""
    try:
        stat = path.stat()
        raw = f"{path.resolve()}::{stat.st_size}::{int(stat.st_mtime)}"
    except OSError:
        raw = str(path)
    return hashlib.sha256(raw.encode()).hexdigest()


def _make_metadata(
    source: str,
    page: int | None = None,
    title: str = "",
    doc_type: str = "",
) -> dict[str, Any]:
    """Create a standardised metadata dict.

    Args:
        source: File path or identifier.
        page: 1-based page number (None for non-paged formats).
        title: Section title or document title.
        doc_type: File type hint (pdf, markdown, text).

    Returns:
        Metadata dictionary with all standard keys.
    """
    return {
        "source": source,
        "page": page,
        "title": title,
        "doc_type": doc_type,
    }


# ---------------------------------------------------------------------------
# Public loader functions
# ---------------------------------------------------------------------------


def load_pdf(path: str | Path) -> list[Document]:
    """Load a PDF file using PyMuPDF, creating one Document per page.

    Each document's ``metadata`` contains ``source`` (file path), ``page``
    (1-based page number), ``title`` (filename), and ``doc_type`` ('pdf').

    Args:
        path: Filesystem path to a PDF file.

    Returns:
        List of ``Document`` instances, one per page.
    """
    path = Path(path)
    cache_k = _cache_key(path)
    if cache_k in _CACHE:
        logger.debug("Returning cached documents for %s", path)
        return _CACHE[cache_k]

    try:
        import fitz  # PyMuPDF  # type: ignore[import-untyped]
    except ImportError:
        logger.error("PyMuPDF (fitz) is not installed – cannot load PDFs")
        return []

    try:
        doc = fitz.open(str(path))
        docs: list[Document] = []
        source = str(path)
        filename = path.stem  # 文件名（不含扩展名）

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text("text")
            if not text or not text.strip():
                continue
            docs.append(
                Document(
                    content=text.strip(),
                    metadata=_make_metadata(
                        source=source,
                        page=page_idx + 1,  # 1-based page number
                        title=filename,
                        doc_type="pdf",
                    ),
                )
            )

        doc.close()
        _CACHE[cache_k] = docs
        logger.info("Loaded %d pages from PDF: %s", len(docs), path)
        return docs
    except Exception:
        logger.exception("Failed to load PDF: %s", path)
        return []


def load_markdown(path: str | Path) -> list[Document]:
    """Load a Markdown file, splitting on header headings (## and ###).

    Each section becomes a separate Document. The header text is stored
    in the ``title`` metadata field. Text before the first header is
    included as chunk 0 with the filename as title.

    Args:
        path: Filesystem path to a Markdown file.

    Returns:
        List of ``Document`` instances, one per section.
    """
    path = Path(path)
    cache_k = _cache_key(path)
    if cache_k in _CACHE:
        logger.debug("Returning cached documents for %s", path)
        return _CACHE[cache_k]

    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        logger.exception("Failed to read Markdown file: %s", path)
        return []

    source = str(path)
    filename = path.stem

    # Split on lines that start with ## (atx-style headers)
    # 正则：匹配 ## 或更高级别的标题行
    sections = re.split(r"\n(?=#{1,3} )", text)
    docs: list[Document] = []

    for section in sections:
        cleaned = section.strip()
        if not cleaned:
            continue

        # 提取标题：取第一个 # 开头的行
        title_match = re.match(r"^(#{1,3})\s+(.+)", cleaned)
        if title_match:
            section_title = title_match.group(2).strip()
        else:
            section_title = filename  # 没有标题时使用文件名

        docs.append(
            Document(
                content=cleaned,
                metadata=_make_metadata(
                    source=source,
                    page=None,
                    title=section_title,
                    doc_type="markdown",
                ),
            )
        )

    if not docs:
        # 整个文件作为一个文档
        docs.append(
            Document(
                content=text.strip(),
                metadata=_make_metadata(
                    source=source,
                    page=None,
                    title=filename,
                    doc_type="markdown",
                ),
            )
        )

    _CACHE[cache_k] = docs
    logger.info("Loaded %d sections from Markdown: %s", len(docs), path)
    return docs


def load_text(path: str | Path) -> list[Document]:
    """Load a plain-text file.

    If the file contains double-newline separated paragraphs, each paragraph
    becomes a separate Document. Otherwise the entire file is returned as a
    single Document.

    Args:
        path: Filesystem path to a text file.

    Returns:
        List of ``Document`` instances.
    """
    path = Path(path)
    cache_k = _cache_key(path)
    if cache_k in _CACHE:
        logger.debug("Returning cached documents for %s", path)
        return _CACHE[cache_k]

    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        logger.exception("Failed to read text file: %s", path)
        return []

    source = str(path)
    filename = path.stem
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if len(paragraphs) > 1:
        docs = [
            Document(
                content=para,
                metadata=_make_metadata(
                    source=source,
                    page=None,
                    title=f"{filename}_para_{idx}",
                    doc_type="text",
                ),
            )
            for idx, para in enumerate(paragraphs)
        ]
    else:
        docs = [
            Document(
                content=text.strip(),
                metadata=_make_metadata(
                    source=source,
                    page=None,
                    title=filename,
                    doc_type="text",
                ),
            )
        ]

    _CACHE[cache_k] = docs
    logger.info("Loaded %d sections from text: %s", len(docs), path)
    return docs


# ---------------------------------------------------------------------------
# Auto-detect and directory loader
# ---------------------------------------------------------------------------

_EXTENSION_MAP: dict[str, Any] = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_text,
    ".text": load_text,
}


def load_file(path: str | Path) -> list[Document]:
    """Auto-detect file type by extension and delegate to the right loader.

    Args:
        path: Filesystem path to a supported file.

    Returns:
        List of ``Document`` instances produced by the appropriate loader.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("File does not exist: %s", p)
        return []

    suffix = p.suffix.lower()
    loader = _EXTENSION_MAP.get(suffix)
    if loader is None:
        logger.warning("Unsupported file extension '%s' for %s", suffix, p)
        return []

    return loader(p)


def load_directory(path: str | Path) -> list[Document]:
    """Recursively load all supported files in a directory.

    Supported extensions: ``.pdf``, ``.md``, ``.markdown``, ``.txt``,
    ``.text``.

    Args:
        path: Filesystem path to a directory.

    Returns:
        Combined list of ``Document`` instances from all supported files.
    """
    dir_path = Path(path)
    if not dir_path.is_dir():
        logger.warning("Not a directory: %s", dir_path)
        return []

    all_docs: list[Document] = []
    supported = set(_EXTENSION_MAP.keys())
    file_count = 0

    for file in sorted(dir_path.rglob("*")):
        if file.is_file() and file.suffix.lower() in supported:
            docs = load_file(file)
            all_docs.extend(docs)
            file_count += 1

    logger.info(
        "Loaded %d documents from %d files in %s",
        len(all_docs),
        file_count,
        dir_path,
    )
    return all_docs


def clear_cache() -> None:
    """Clear the in-memory document cache."""
    _CACHE.clear()
    logger.debug("Document cache cleared")
