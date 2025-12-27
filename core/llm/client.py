"""Claude API client for code understanding.

This module provides an async client for the Claude API with
streaming support for code-related queries.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, Field

from core.llm.context import CodeContext, TokenCounter
from core.llm.prompts import QueryType, get_prompt_template

logger = structlog.get_logger(__name__)


class LLMClientError(Exception):
    """Base exception for LLM client errors."""

    pass


class RateLimitError(LLMClientError):
    """Rate limit exceeded."""

    pass


class TokenLimitError(LLMClientError):
    """Token limit exceeded."""

    pass


class LLMConfig(BaseModel):
    """Configuration for LLM client.

    Attributes:
        model: Model identifier.
        max_tokens: Maximum output tokens.
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.
    """

    model_config = ConfigDict(frozen=True)

    model: str = Field(default="claude-sonnet-4-20250514", description="Model ID")
    max_tokens: int = Field(default=4096, ge=1, le=8192, description="Max output tokens")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Temperature")
    timeout: float = Field(default=120.0, description="Timeout in seconds")
    max_retries: int = Field(default=3, ge=1, le=10, description="Max retries")


class LLMResponse(BaseModel):
    """Response from LLM.

    Attributes:
        content: The response text.
        model: Model that generated the response.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        stop_reason: Reason for stopping.
        query_type: Type of query this responds to.
    """

    model_config = ConfigDict(frozen=True)

    content: str = Field(..., description="Response content")
    model: str = Field(..., description="Model used")
    input_tokens: int = Field(default=0, description="Input tokens")
    output_tokens: int = Field(default=0, description="Output tokens")
    stop_reason: str = Field(default="end_turn", description="Stop reason")
    query_type: QueryType | None = Field(None, description="Query type")


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def complete(
        self,
        system: str,
        user: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion.

        Args:
            system: System message.
            user: User message.
            **kwargs: Additional parameters.

        Returns:
            LLM response.
        """
        pass

    @abstractmethod
    def stream(
        self,
        system: str,
        user: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion.

        Args:
            system: System message.
            user: User message.
            **kwargs: Additional parameters.

        Yields:
            Response text chunks.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the client and release resources."""
        pass


class ClaudeClient(LLMClient):
    """Async client for Claude API.

    This client provides methods for generating completions
    and streaming responses from the Claude API.
    """

    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str,
        config: LLMConfig | None = None,
    ) -> None:
        """Initialize Claude client.

        Args:
            api_key: Anthropic API key.
            config: Client configuration.
        """
        self.api_key = api_key
        self.config = config or LLMConfig()
        self._client = httpx.AsyncClient(
            timeout=self.config.timeout,
            headers={
                "x-api-key": api_key,
                "anthropic-version": self.API_VERSION,
                "content-type": "application/json",
            },
        )
        self._token_counter = TokenCounter()
        self._logger = logger.bind(component="claude_client")

    async def complete(
        self,
        system: str,
        user: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion from Claude.

        Args:
            system: System message.
            user: User message.
            **kwargs: Additional parameters (temperature, max_tokens, etc.).

        Returns:
            LLM response.

        Raises:
            LLMClientError: If the request fails.
        """
        payload = self._build_payload(system, user, stream=False, **kwargs)

        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.post(self.API_URL, json=payload)

                if response.status_code == 429:
                    # Rate limited
                    wait_time = 2**attempt
                    self._logger.warning("rate_limited", attempt=attempt, wait_time=wait_time)
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code != 200:
                    error_body = response.text
                    raise LLMClientError(f"Claude API error: {response.status_code} - {error_body}")

                data = response.json()
                return self._parse_response(data)

            except httpx.TimeoutException:
                self._logger.warning("timeout", attempt=attempt)
                if attempt == self.config.max_retries - 1:
                    raise LLMClientError("Claude API timeout")
                await asyncio.sleep(2**attempt)

        raise LLMClientError("Max retries exceeded")

    async def stream(
        self,
        system: str,
        user: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion from Claude.

        Args:
            system: System message.
            user: User message.
            **kwargs: Additional parameters.

        Yields:
            Response text chunks.

        Raises:
            LLMClientError: If the request fails.
        """
        payload = self._build_payload(system, user, stream=True, **kwargs)

        try:
            async with self._client.stream(
                "POST",
                self.API_URL,
                json=payload,
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    raise LLMClientError(
                        f"Claude API error: {response.status_code} - {error_body.decode()}"
                    )

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            import json

                            event = json.loads(data)
                            if event.get("type") == "content_block_delta":
                                delta = event.get("delta", {})
                                text = delta.get("text", "")
                                if text:
                                    yield text
                        except Exception:
                            continue

        except httpx.TimeoutException:
            raise LLMClientError("Claude API timeout during streaming")

    async def query(
        self,
        question: str,
        context: CodeContext | None = None,
        query_type: QueryType | None = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[str]:
        """Query Claude with code context.

        This is a high-level method that handles prompt formatting.

        Args:
            question: The user's question.
            context: Code context to include.
            query_type: Type of query (auto-detected if not provided).
            stream: Whether to stream the response.

        Returns:
            LLM response or async iterator for streaming.
        """
        from core.llm.prompts import classify_query

        # Auto-classify query type if not provided
        if query_type is None:
            query_type = classify_query(question)

        # Get appropriate prompt template
        template = get_prompt_template(query_type)

        # Format context
        context_str = context.format() if context else ""

        # Build messages
        system = template.system_prompt
        user = template.format_user_message(
            question=question,
            context=context_str,
        )

        if stream:
            return self.stream(system, user)
        else:
            response = await self.complete(system, user)
            return LLMResponse(
                content=response.content,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                stop_reason=response.stop_reason,
                query_type=query_type,
            )

    def _build_payload(
        self,
        system: str,
        user: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build API request payload.

        Args:
            system: System message.
            user: User message.
            stream: Whether to stream.
            **kwargs: Additional parameters.

        Returns:
            Request payload dictionary.
        """
        return {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "stream": stream,
        }

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        """Parse API response.

        Args:
            data: Response data from API.

        Returns:
            Parsed LLM response.
        """
        content_blocks = data.get("content", [])
        content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                content += block.get("text", "")

        usage = data.get("usage", {})
        return LLMResponse(
            content=content,
            model=data.get("model", self.config.model),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            stop_reason=data.get("stop_reason", "end_turn"),
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


class MockClaudeClient(LLMClient):
    """Mock Claude client for testing.

    Returns predefined responses without making API calls.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        """Initialize mock client.

        Args:
            responses: List of responses to return in order.
        """
        self._responses = responses or ["This is a mock response."]
        self._call_count = 0
        self._calls: list[dict[str, Any]] = []

    async def complete(
        self,
        system: str,
        user: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """Return a mock completion.

        Args:
            system: System message.
            user: User message.
            **kwargs: Additional parameters.

        Returns:
            Mock LLM response.
        """
        self._calls.append({"system": system, "user": user, **kwargs})
        response_text = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1

        return LLMResponse(
            content=response_text,
            model="mock-model",
            input_tokens=len(system) + len(user),
            output_tokens=len(response_text),
            stop_reason="end_turn",
        )

    async def stream(
        self,
        system: str,
        user: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a mock completion.

        Args:
            system: System message.
            user: User message.
            **kwargs: Additional parameters.

        Yields:
            Response text chunks.
        """
        self._calls.append({"system": system, "user": user, **kwargs})
        response_text = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1

        # Yield word by word for realistic streaming simulation
        words = response_text.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.01)

    async def close(self) -> None:
        """No-op for mock client."""
        pass

    async def query(
        self,
        question: str,
        context: CodeContext | None = None,
        query_type: QueryType | None = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[str]:
        """Query with code context (mock implementation).

        Args:
            question: The user's question.
            context: Code context to include.
            query_type: Type of query (auto-detected if not provided).
            stream: Whether to stream the response.

        Returns:
            LLM response or async iterator for streaming.
        """
        from core.llm.prompts import classify_query

        # Auto-classify query type if not provided
        if query_type is None:
            query_type = classify_query(question)

        # Get appropriate prompt template
        template = get_prompt_template(query_type)

        # Format context
        context_str = context.format() if context else ""

        # Build messages
        system = template.system_prompt
        user = template.format_user_message(
            question=question,
            context=context_str,
        )

        if stream:
            return self.stream(system, user)
        else:
            response = await self.complete(system, user)
            return LLMResponse(
                content=response.content,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                stop_reason=response.stop_reason,
                query_type=query_type,
            )

    @property
    def calls(self) -> list[dict[str, Any]]:
        """Get list of calls made to this client."""
        return self._calls

    def reset(self) -> None:
        """Reset call history."""
        self._calls = []
        self._call_count = 0
