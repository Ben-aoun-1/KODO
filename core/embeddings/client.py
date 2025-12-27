"""Embedding client for generating vector embeddings.

This module provides clients for Voyage AI and OpenAI embedding APIs.
"""

import asyncio
from abc import ABC, abstractmethod

import httpx
import structlog
import tiktoken

from core.embeddings.models import (
    OPENAI_CONFIG,
    VOYAGE_CODE_CONFIG,
    EmbeddingBatchResult,
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingResult,
)

logger = structlog.get_logger(__name__)


class EmbeddingClientError(Exception):
    """Base exception for embedding client errors."""

    pass


class RateLimitError(EmbeddingClientError):
    """Rate limit exceeded."""

    pass


class TokenLimitError(EmbeddingClientError):
    """Token limit exceeded."""

    pass


class EmbeddingClient(ABC):
    """Abstract base class for embedding clients."""

    @abstractmethod
    async def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding result.
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> EmbeddingBatchResult:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            Batch embedding result.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the client and release resources."""
        pass


class VoyageClient(EmbeddingClient):
    """Client for Voyage AI embeddings.

    Voyage AI provides specialized code embeddings optimized for
    semantic code search and understanding.
    """

    API_URL = "https://api.voyageai.com/v1/embeddings"

    def __init__(
        self,
        api_key: str,
        config: EmbeddingConfig | None = None,
    ) -> None:
        """Initialize Voyage client.

        Args:
            api_key: Voyage AI API key.
            config: Embedding configuration.
        """
        self.api_key = api_key
        self.config = config or VOYAGE_CODE_CONFIG
        self._client = httpx.AsyncClient(
            timeout=60.0,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        self._logger = logger.bind(provider="voyage")

    async def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding result.

        Raises:
            EmbeddingClientError: If embedding fails.
        """
        result = await self.embed_batch([text])
        if result.embeddings:
            return result.embeddings[0]
        raise EmbeddingClientError("Failed to generate embedding")

    async def embed_batch(self, texts: list[str]) -> EmbeddingBatchResult:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            Batch embedding result.

        Raises:
            EmbeddingClientError: If embedding fails.
        """
        if not texts:
            return EmbeddingBatchResult(embeddings=[], model=self.config.model)

        embeddings: list[EmbeddingResult] = []
        failed_indices: list[int] = []
        total_tokens = 0

        # Process in batches
        for batch_start in range(0, len(texts), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]

            try:
                batch_result = await self._embed_batch_internal(batch_texts)
                for text, vector, tokens in batch_result:
                    embeddings.append(
                        EmbeddingResult(
                            text=text,
                            vector=vector,
                            model=self.config.model,
                            token_count=tokens,
                        )
                    )
                    total_tokens += tokens
            except EmbeddingClientError:
                # Mark all texts in this batch as failed
                for i in range(batch_start, batch_end):
                    failed_indices.append(i)
                self._logger.error(
                    "batch_embedding_failed",
                    batch_start=batch_start,
                    batch_end=batch_end,
                )

        return EmbeddingBatchResult(
            embeddings=embeddings,
            total_tokens=total_tokens,
            model=self.config.model,
            failed_indices=failed_indices,
        )

    async def _embed_batch_internal(
        self,
        texts: list[str],
    ) -> list[tuple[str, list[float], int]]:
        """Internal method to embed a batch of texts.

        Args:
            texts: Batch of texts.

        Returns:
            List of (text, vector, token_count) tuples.

        Raises:
            EmbeddingClientError: If API call fails.
        """
        payload = {
            "input": texts,
            "model": self.config.model,
            "input_type": "document",
        }

        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.post(self.API_URL, json=payload)

                if response.status_code == 429:
                    # Rate limited - exponential backoff
                    wait_time = 2**attempt
                    self._logger.warning(
                        "rate_limited",
                        attempt=attempt,
                        wait_time=wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code != 200:
                    raise EmbeddingClientError(
                        f"Voyage API error: {response.status_code} - {response.text}"
                    )

                data = response.json()
                results = []
                for i, item in enumerate(data.get("data", [])):
                    vector = item.get("embedding", [])
                    # Voyage doesn't return per-text token counts
                    results.append((texts[i], vector, 0))

                # Update total token count from usage
                usage = data.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)
                if results and total_tokens:
                    # Distribute tokens roughly evenly
                    per_text = total_tokens // len(results)
                    results = [(t, v, per_text) for t, v, _ in results]

                return results

            except httpx.TimeoutException:
                self._logger.warning("timeout", attempt=attempt)
                if attempt == self.config.max_retries - 1:
                    raise EmbeddingClientError("Voyage API timeout")
                await asyncio.sleep(2**attempt)

        raise EmbeddingClientError("Max retries exceeded")

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


class OpenAIClient(EmbeddingClient):
    """Client for OpenAI embeddings.

    OpenAI provides general-purpose text embeddings that work well
    for code when code-specific models aren't available.
    """

    API_URL = "https://api.openai.com/v1/embeddings"

    def __init__(
        self,
        api_key: str,
        config: EmbeddingConfig | None = None,
    ) -> None:
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key.
            config: Embedding configuration.
        """
        self.api_key = api_key
        self.config = config or OPENAI_CONFIG
        self._client = httpx.AsyncClient(
            timeout=60.0,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        self._logger = logger.bind(provider="openai")
        # Use cl100k_base for OpenAI embeddings
        try:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for.

        Returns:
            Token count.
        """
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # Rough estimate if tokenizer unavailable
        return len(text) // 4

    async def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding result.

        Raises:
            EmbeddingClientError: If embedding fails.
        """
        result = await self.embed_batch([text])
        if result.embeddings:
            return result.embeddings[0]
        raise EmbeddingClientError("Failed to generate embedding")

    async def embed_batch(self, texts: list[str]) -> EmbeddingBatchResult:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            Batch embedding result.

        Raises:
            EmbeddingClientError: If embedding fails.
        """
        if not texts:
            return EmbeddingBatchResult(embeddings=[], model=self.config.model)

        embeddings: list[EmbeddingResult] = []
        failed_indices: list[int] = []
        total_tokens = 0

        # Process in batches
        for batch_start in range(0, len(texts), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]

            try:
                batch_result = await self._embed_batch_internal(batch_texts)
                for text, vector, tokens in batch_result:
                    embeddings.append(
                        EmbeddingResult(
                            text=text,
                            vector=vector,
                            model=self.config.model,
                            token_count=tokens,
                        )
                    )
                    total_tokens += tokens
            except EmbeddingClientError:
                for i in range(batch_start, batch_end):
                    failed_indices.append(i)
                self._logger.error(
                    "batch_embedding_failed",
                    batch_start=batch_start,
                    batch_end=batch_end,
                )

        return EmbeddingBatchResult(
            embeddings=embeddings,
            total_tokens=total_tokens,
            model=self.config.model,
            failed_indices=failed_indices,
        )

    async def _embed_batch_internal(
        self,
        texts: list[str],
    ) -> list[tuple[str, list[float], int]]:
        """Internal method to embed a batch of texts.

        Args:
            texts: Batch of texts.

        Returns:
            List of (text, vector, token_count) tuples.

        Raises:
            EmbeddingClientError: If API call fails.
        """
        payload = {
            "input": texts,
            "model": self.config.model,
        }

        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.post(self.API_URL, json=payload)

                if response.status_code == 429:
                    wait_time = 2**attempt
                    self._logger.warning(
                        "rate_limited",
                        attempt=attempt,
                        wait_time=wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code != 200:
                    raise EmbeddingClientError(
                        f"OpenAI API error: {response.status_code} - {response.text}"
                    )

                data = response.json()
                results = []

                # Sort by index to maintain order
                items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
                for i, item in enumerate(items):
                    vector = item.get("embedding", [])
                    token_count = self.count_tokens(texts[i])
                    results.append((texts[i], vector, token_count))

                return results

            except httpx.TimeoutException:
                self._logger.warning("timeout", attempt=attempt)
                if attempt == self.config.max_retries - 1:
                    raise EmbeddingClientError("OpenAI API timeout")
                await asyncio.sleep(2**attempt)

        raise EmbeddingClientError("Max retries exceeded")

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


def create_embedding_client(
    provider: EmbeddingProvider,
    api_key: str,
    config: EmbeddingConfig | None = None,
) -> EmbeddingClient:
    """Create an embedding client for the specified provider.

    Args:
        provider: Embedding provider.
        api_key: API key for the provider.
        config: Optional configuration override.

    Returns:
        Configured embedding client.

    Raises:
        ValueError: If provider is not supported.
    """
    if provider == EmbeddingProvider.VOYAGE:
        return VoyageClient(api_key, config)
    elif provider == EmbeddingProvider.OPENAI:
        return OpenAIClient(api_key, config)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
