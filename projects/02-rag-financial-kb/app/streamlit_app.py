"""
Streamlit frontend for the RAG Financial Knowledge Base.

Provides a chat-like interface for asking financial questions,
a source display panel, and a document upload sidebar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

API_BASE = st.secrets.get("API_BASE", "http://localhost:8000") if hasattr(st, "secrets") else "http://localhost:8000"

# Try environment variable first
import os

API_BASE = os.getenv("API_BASE", API_BASE)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Financial KB",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .source-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.375rem;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
    }
    .source-card .score {
        color: #0d6efd;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def check_api_health() -> bool:
    """Check if the API server is reachable."""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def query_rag(question: str, top_k: int = 5, use_reranker: bool = True) -> dict:
    """Send a query to the RAG API."""
    resp = requests.post(
        f"{API_BASE}/query",
        json={"question": question, "top_k": top_k, "use_reranker": use_reranker},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def ingest_file(file) -> dict:
    """Upload a file to the ingest endpoint."""
    resp = requests.post(
        f"{API_BASE}/ingest",
        files={"file": (file.name, file.getvalue())},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def ingest_path(path: str) -> dict:
    """Ingest documents from a filesystem path."""
    resp = requests.post(
        f"{API_BASE}/ingest",
        json={"path": path, "strategy": "recursive"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def get_stats() -> dict:
    """Get vector store stats from the API."""
    resp = requests.get(f"{API_BASE}/stats", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    """Render the document upload sidebar."""
    with st.sidebar:
        st.header("📚 Document Management")

        # API status
        api_ok = check_api_health()
        if api_ok:
            st.success("✅ API connected")
        else:
            st.error("❌ API unreachable")
            st.info(f"Expected at: `{API_BASE}`")

        st.divider()

        # File upload
        st.subheader("Upload File")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "md", "txt", "html", "htm", "jsonl"],
            help="Supported: PDF, Markdown, TXT, HTML, JSONL",
        )

        if uploaded_file and st.button("📤 Ingest Uploaded File", type="primary"):
            with st.spinner("Uploading and ingesting..."):
                try:
                    result = ingest_file(uploaded_file)
                    st.success(
                        f"✅ Ingested: {result['documents_loaded']} docs → "
                        f"{result['chunks_created']} chunks → "
                        f"{result['chunks_indexed']} indexed"
                    )
                except requests.HTTPError as e:
                    st.error(f"Ingest failed: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()

        # Directory path ingestion
        st.subheader("Ingest from Path")
        dir_path = st.text_input(
            "Directory or file path",
            placeholder="/path/to/documents/",
            help="Enter a filesystem path to ingest all supported files",
        )

        if dir_path and st.button("📁 Ingest from Path", type="secondary"):
            with st.spinner("Loading and ingesting documents..."):
                try:
                    result = ingest_path(dir_path)
                    st.success(
                        f"✅ Ingested: {result['documents_loaded']} docs → "
                        f"{result['chunks_created']} chunks → "
                        f"{result['chunks_indexed']} indexed"
                    )
                except requests.HTTPError as e:
                    st.error(f"Ingest failed: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()

        # Stats
        st.subheader("📊 Stats")
        if st.button("Refresh Stats"):
            try:
                stats = get_stats()
                st.metric("Total Documents", stats["total_documents"])
                st.caption(f"Embedding Provider: {stats['embedding_provider']}")
                cache = stats.get("embedding_cache_stats", {})
                if cache:
                    st.caption(
                        f"Cache: {cache.get('size', 0)} entries "
                        f"({cache.get('hits', 0)} hits, {cache.get('misses', 0)} misses)"
                    )
            except Exception as e:
                st.error(f"Failed to load stats: {e}")

        st.divider()

        # Settings
        st.subheader("⚙️ Settings")
        st.text_input("API Base URL", value=API_BASE, key="api_url_display", disabled=True)

        # Query settings
        if "top_k" not in st.session_state:
            st.session_state.top_k = 5
        if "use_reranker" not in st.session_state:
            st.session_state.use_reranker = True

        st.session_state.top_k = st.slider(
            "Top-K Results", min_value=1, max_value=20, value=st.session_state.top_k
        )
        st.session_state.use_reranker = st.checkbox(
            "Use Reranker", value=st.session_state.use_reranker
        )


# ---------------------------------------------------------------------------
# Source display


def render_sources(sources: list[dict]) -> None:
    """Render source documents in an expandable panel."""
    if not sources:
        return

    with st.expander(f"📎 Sources ({len(sources)} documents)", expanded=False):
        for i, source in enumerate(sources):
            score = source.get("score", 0.0)
            text = source.get("text", "")
            metadata = source.get("metadata", {})

            source_label = metadata.get("source", "Unknown")
            page = metadata.get("page")
            page_str = f" — Page {page}" if page else ""

            st.markdown(
                f"""
                <div class="source-card">
                    <div><strong>[{i + 1}]</strong> {source_label}{page_str}</div>
                    <div class="score">Score: {score:.4f}</div>
                    <div style="margin-top: 0.25rem; color: #495057;">{text[:500]}{'...' if len(text) > 500 else ''}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Main chat interface


def main() -> None:
    """Main Streamlit application."""
    st.title("📊 RAG Financial Knowledge Base")
    st.caption("Ask questions about financial documents — powered by retrieval-augmented generation")

    # Render sidebar
    render_sidebar()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])
            if message.get("model"):
                st.caption(f"Model: {message['model']}")

    # Chat input
    if prompt := st.chat_input("Ask a financial question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Check API health first
        if not check_api_health():
            with st.chat_message("assistant"):
                st.error("Cannot connect to the RAG API. Please check the server.")
                st.info(f"Expected at: `{API_BASE}`")
            return

        # Query RAG
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base and generating answer..."):
                try:
                    result = query_rag(
                        question=prompt,
                        top_k=st.session_state.top_k,
                        use_reranker=st.session_state.use_reranker,
                    )

                    st.markdown(result["answer"])
                    render_sources(result.get("sources", []))
                    st.caption(f"Model: {result.get('model', 'unknown')}")

                    # Save to history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result.get("sources", []),
                            "model": result.get("model", ""),
                        }
                    )

                except requests.HTTPError as e:
                    error_msg = f"API error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

    # Clear history button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
