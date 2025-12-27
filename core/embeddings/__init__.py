"""Embeddings module for Kodo.

This module provides vector embedding functionality for semantic code search,
including chunking strategies, embedding generation, and Qdrant vector store.
"""

from core.embeddings.chunker import ChunkerConfig, ChunkType, CodeChunk, CodeChunker
from core.embeddings.client import (
    EmbeddingClient,
    EmbeddingClientError,
    OpenAIClient,
    RateLimitError,
    TokenLimitError,
    VoyageClient,
    create_embedding_client,
)
from core.embeddings.models import (
    OPENAI_CONFIG,
    OPENAI_TEXT_EMBEDDING_3_LARGE,
    OPENAI_TEXT_EMBEDDING_3_SMALL,
    VOYAGE_CODE_2,
    VOYAGE_CODE_CONFIG,
    EmbeddingBatchResult,
    EmbeddingConfig,
    EmbeddingModel,
    EmbeddingProvider,
    EmbeddingResult,
)
from core.embeddings.search import CodeSearchResult, SemanticSearch, SemanticSearchConfig
from core.embeddings.store import (
    CollectionNotFoundError,
    SearchResult,
    VectorStore,
    VectorStoreConfig,
    VectorStoreError,
)

__all__ = [
    # Chunker
    "ChunkType",
    "CodeChunk",
    "CodeChunker",
    "ChunkerConfig",
    # Client
    "EmbeddingClient",
    "EmbeddingClientError",
    "OpenAIClient",
    "RateLimitError",
    "TokenLimitError",
    "VoyageClient",
    "create_embedding_client",
    # Models
    "EmbeddingBatchResult",
    "EmbeddingConfig",
    "EmbeddingModel",
    "EmbeddingProvider",
    "EmbeddingResult",
    "OPENAI_CONFIG",
    "OPENAI_TEXT_EMBEDDING_3_LARGE",
    "OPENAI_TEXT_EMBEDDING_3_SMALL",
    "VOYAGE_CODE_2",
    "VOYAGE_CODE_CONFIG",
    # Search
    "CodeSearchResult",
    "SemanticSearch",
    "SemanticSearchConfig",
    # Store
    "CollectionNotFoundError",
    "SearchResult",
    "VectorStore",
    "VectorStoreConfig",
    "VectorStoreError",
]
