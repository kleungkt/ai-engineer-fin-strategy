"""
RAG Financial Knowledge Base – core modules.

Submodules:
    document_loader: Load documents from PDF, Markdown, and text files.
    text_splitter: Split documents into chunks with multiple strategies.
    embedding: OpenAI and local embedding model wrappers.
    vector_store: ChromaDB-backed vector storage.
    retriever: Similarity, MMR, and hybrid search strategies.
    rag_engine: Core RAG pipeline with query rewriting.
    source_tracker: Citation extraction and validation.
    evaluator: RAG quality evaluation (Precision@K, MRR, faithfulness).
"""

from __future__ import annotations
