"""
CLI entry point for the RAG financial knowledge base.

Usage::

    python main.py ingest --dir data/sample_docs/
    python main.py query --q "什麼是MACD"
    python main.py interactive
    python main.py demo
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.document_loader import load_directory
from src.text_splitter import split_documents
from src.embedding import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.rag_engine import RAGEngine, RAGResponse
from src.source_tracker import SourceTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: build components from settings
# ---------------------------------------------------------------------------


def _build_components() -> tuple[VectorStore, EmbeddingModel, Retriever, RAGEngine]:
    """Create and wire up all RAG pipeline components.

    Returns:
        Tuple of (vector_store, embedding_model, retriever, rag_engine).
    """
    from config.settings import settings

    embedding_model = EmbeddingModel(model_name=settings.EMBEDDING_MODEL)
    vector_store = VectorStore(
        persist_dir=settings.CHROMA_PERSIST_DIR,
        collection_name="financial_kb",
    )
    retriever = Retriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
    )
    rag_engine = RAGEngine(
        retriever=retriever,
        llm_model=settings.OPENAI_MODEL,
    )
    return vector_store, embedding_model, retriever, rag_engine


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def ingest_command(directory: str) -> None:
    """Load documents from *directory*, split, embed, and store them.

    Args:
        directory: Path to a directory containing documents.
    """
    from config.settings import settings

    dir_path = Path(directory)
    if not dir_path.is_dir():
        print(f"❌ 目录不存在: {directory}")
        return

    print(f"📂 正在加载文档: {directory}")
    docs = load_directory(dir_path)
    if not docs:
        print("⚠️  未找到可加载的文档。")
        return
    print(f"   加载了 {len(docs)} 个文档段落")

    print("✂️  正在分割文档...")
    chunks = split_documents(
        docs,
        strategy="recursive",
        chunk_size=settings.CHUNK_SIZE,
        overlap=settings.CHUNK_OVERLAP,
    )
    print(f"   生成了 {len(chunks)} 个文本块")

    print("🔢 正在生成 embeddings...")
    _, embedding_model, _, _ = _build_components()
    texts = [c.content for c in chunks]
    embeddings = embedding_model.embed_texts(texts)

    print("💾 正在写入向量数据库...")
    vector_store = VectorStore(
        persist_dir=settings.CHROMA_PERSIST_DIR,
        collection_name="financial_kb",
    )
    added = vector_store.add_chunks(chunks, embeddings)
    vector_store.persist()

    print(f"✅ 完成! 共写入 {added} 个文本块到知识库。")


def query_command(question: str) -> None:
    """Run a RAG query and print the formatted answer.

    Args:
        question: The question to ask.
    """
    _, _, retriever, rag_engine = _build_components()

    print(f"\n🔍 查询: {question}")
    print("=" * 60)

    response: RAGResponse = rag_engine.query(question, top_k=5, use_rewrite=True)

    if response.rewritten_query and response.rewritten_query != question:
        print(f"\n📝 改写查询: {response.rewritten_query}")

    print(f"\n💡 回答 (置信度: {response.confidence:.2f}):\n")
    print(response.answer)

    # 展示来源
    if response.sources:
        tracker = SourceTracker()
        # 转换为 tracker 格式
        source_chunks = [
            {"content": s["content"], "metadata": {"source": s.get("source", "")}, "score": s.get("score", 0.0)}
            for s in response.sources
        ]
        print(f"\n{tracker.format_sources(source_chunks)}")


def interactive() -> None:
    """Start an interactive REPL for querying the knowledge base."""
    print("\n" + "=" * 60)
    print("  金融知識庫 RAG 系統 — 互動模式")
    print("  輸入問題進行查詢，輸入 'quit' 或 'exit' 退出")
    print("=" * 60 + "\n")

    _, _, _, rag_engine = _build_components()

    while True:
        try:
            question = input("❓ 請輸入問題: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再見!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("👋 再見!")
            break

        try:
            response = rag_engine.query(question, top_k=5, use_rewrite=True)
            print(f"\n💡 回答 (置信度: {response.confidence:.2f}):\n")
            print(response.answer)

            if response.sources:
                print(f"\n📚 引用了 {len(response.sources)} 个来源")
                for i, src in enumerate(response.sources, 1):
                    source_name = src.get("source", "未知")
                    score = src.get("score", 0.0)
                    print(f"   [{i}] {source_name} (相关度: {score:.3f})")

        except Exception as exc:
            print(f"❌ 查询出错: {exc}")

        print()  # 空行分隔


def demo() -> None:
    """Run preset queries to demonstrate the RAG system."""
    demo_queries = [
        "什麼是MACD指標？",
        "如何使用布林帶進行交易？",
        "風險管理的基本原則是什麼？",
        "什麼是移動平均線交叉策略？",
    ]

    print("\n" + "=" * 60)
    print("  金融知識庫 RAG 系統 — 示範模式")
    print("=" * 60)

    _, _, _, rag_engine = _build_components()

    for i, question in enumerate(demo_queries, 1):
        print(f"\n{'─' * 60}")
        print(f"  示範 {i}/{len(demo_queries)}: {question}")
        print(f"{'─' * 60}")

        try:
            response = rag_engine.query(question, top_k=3, use_rewrite=False)
            print(f"\n💡 回答 (置信度: {response.confidence:.2f}):\n")
            print(response.answer)

            if response.sources:
                print(f"\n📚 引用了 {len(response.sources)} 个来源")
        except Exception as exc:
            print(f"❌ 查询出错: {exc}")

    print(f"\n{'=' * 60}")
    print("  示範完成!")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Argparse CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(
        description="RAG 金融知識庫 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
使用示例:
  python main.py ingest --dir data/sample_docs/
  python main.py query --q "什麼是MACD"
  python main.py interactive
  python main.py demo
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # ingest 子命令
    ingest_parser = subparsers.add_parser("ingest", help="加载文档到知识库")
    ingest_parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="文档目录路径",
    )

    # query 子命令
    query_parser = subparsers.add_parser("query", help="查询知识库")
    query_parser.add_argument(
        "--q",
        type=str,
        required=True,
        help="查询问题",
    )

    # interactive 子命令
    subparsers.add_parser("interactive", help="启动交互式查询")

    # demo 子命令
    subparsers.add_parser("demo", help="运行示范查询")

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.command is None:
        parser.print_help()
        return

    if args.command == "ingest":
        ingest_command(args.dir)
    elif args.command == "query":
        query_command(args.q)
    elif args.command == "interactive":
        interactive()
    elif args.command == "demo":
        demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
