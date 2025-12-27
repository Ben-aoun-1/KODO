"""Embedding models and types for Kodo.

This module defines the data models for embedding generation,
including providers, configurations, and result types.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    VOYAGE = "voyage"
    OPENAI = "openai"


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation.

    Attributes:
        provider: The embedding provider to use.
        model: The model name/ID.
        dimension: Vector dimension size.
        batch_size: Maximum batch size for API calls.
        max_retries: Maximum number of retry attempts.
    """

    model_config = ConfigDict(frozen=True)

    provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.VOYAGE,
        description="Embedding provider",
    )
    model: str = Field(
        default="voyage-code-2",
        description="Model name",
    )
    dimension: int = Field(
        default=1024,
        description="Vector dimension",
    )
    batch_size: int = Field(
        default=128,
        ge=1,
        le=256,
        description="Maximum batch size",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts",
    )


# Preset configurations for common providers
VOYAGE_CODE_CONFIG = EmbeddingConfig(
    provider=EmbeddingProvider.VOYAGE,
    model="voyage-code-2",
    dimension=1024,
    batch_size=128,
)

OPENAI_CONFIG = EmbeddingConfig(
    provider=EmbeddingProvider.OPENAI,
    model="text-embedding-3-large",
    dimension=3072,
    batch_size=100,
)


class EmbeddingResult(BaseModel):
    """Result of embedding generation.

    Attributes:
        text: Original text that was embedded.
        vector: The embedding vector.
        model: Model used for generation.
        token_count: Number of tokens in the text.
        metadata: Additional metadata.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str = Field(..., description="Original text")
    vector: list[float] = Field(..., description="Embedding vector")
    model: str = Field(..., description="Model used")
    token_count: int = Field(default=0, description="Token count")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")


class EmbeddingBatchResult(BaseModel):
    """Result of batch embedding generation.

    Attributes:
        embeddings: List of embedding results.
        total_tokens: Total tokens processed.
        model: Model used for generation.
        failed_indices: Indices of failed texts.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeddings: list[EmbeddingResult] = Field(default_factory=list, description="Embeddings")
    total_tokens: int = Field(default=0, description="Total tokens")
    model: str = Field(..., description="Model used")
    failed_indices: list[int] = Field(default_factory=list, description="Failed indices")


class EmbeddingModel(BaseModel):
    """Base class for embedding model information.

    Attributes:
        provider: The embedding provider.
        model_name: Model identifier.
        dimension: Output vector dimension.
        max_tokens: Maximum input tokens.
        supports_batching: Whether batching is supported.
    """

    model_config = ConfigDict(frozen=True)

    provider: EmbeddingProvider = Field(..., description="Provider")
    model_name: str = Field(..., description="Model name")
    dimension: int = Field(..., description="Vector dimension")
    max_tokens: int = Field(default=8192, description="Max input tokens")
    supports_batching: bool = Field(default=True, description="Supports batching")


# Known embedding models
VOYAGE_CODE_2 = EmbeddingModel(
    provider=EmbeddingProvider.VOYAGE,
    model_name="voyage-code-2",
    dimension=1024,
    max_tokens=16000,
    supports_batching=True,
)

OPENAI_TEXT_EMBEDDING_3_LARGE = EmbeddingModel(
    provider=EmbeddingProvider.OPENAI,
    model_name="text-embedding-3-large",
    dimension=3072,
    max_tokens=8191,
    supports_batching=True,
)

OPENAI_TEXT_EMBEDDING_3_SMALL = EmbeddingModel(
    provider=EmbeddingProvider.OPENAI,
    model_name="text-embedding-3-small",
    dimension=1536,
    max_tokens=8191,
    supports_batching=True,
)
