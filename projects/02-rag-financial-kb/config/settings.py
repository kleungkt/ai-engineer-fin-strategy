"""
Configuration settings for the RAG financial knowledge base system.

This module defines all configurable parameters using Pydantic BaseSettings,
which automatically loads values from environment variables and .env files.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file.

    All settings can be overridden by setting the corresponding environment
    variable with the same name (case-insensitive).
    """

    # OpenAI API settings
    OPENAI_API_KEY: str = Field(
        default="",
        description="OpenAI API key for LLM and embedding services"
    )
    OPENAI_MODEL: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name for chat completions"
    )

    # Embedding model settings
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model for vector representations"
    )

    # ChromaDB settings
    CHROMA_PERSIST_DIR: str = Field(
        default="./chroma_db",
        description="Directory path for ChromaDB persistent storage"
    )

    # Text chunking settings
    CHUNK_SIZE: int = Field(
        default=500,
        description="Maximum number of characters per text chunk"
    )
    CHUNK_OVERLAP: int = Field(
        default=50,
        description="Number of overlapping characters between adjacent chunks"
    )

    # Retrieval settings
    TOP_K: int = Field(
        default=5,
        description="Number of top similar documents to retrieve"
    )
    SIMILARITY_THRESHOLD: float = Field(
        default=0.7,
        description="Minimum similarity score threshold for document retrieval"
    )

    # Reranking settings
    RERANK_ENABLED: bool = Field(
        default=True,
        description="Whether to enable reranking of retrieved documents"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore"
    }


# Global settings instance
settings = Settings()
