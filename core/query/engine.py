"""Query engine for orchestrating code queries.

This module provides the main QueryEngine class that coordinates
query routing, context building, and LLM interaction.
"""

import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field

from core.llm.client import LLMClient, LLMResponse, MockClaudeClient
from core.llm.context import CodeContext, ContextBuilder, ContextConfig
from core.llm.prompts import get_prompt_template
from core.query.handlers import BaseHandler, HandlerContext, get_handler
from core.query.models import (
    QueryRequest,
    QueryResponse,
    QueryStatus,
)
from core.query.router import QueryRouter

logger = structlog.get_logger(__name__)


class QueryEngineConfig(BaseModel):
    """Configuration for the query engine.

    Attributes:
        max_context_tokens: Maximum tokens for context.
        max_response_tokens: Maximum tokens for response.
        include_sources: Default for including source references.
        min_relevance: Minimum relevance score for context.
        max_snippets: Maximum number of snippets in context.
        timeout: Query timeout in seconds.
    """

    model_config = ConfigDict(frozen=True)

    max_context_tokens: int = Field(
        default=8000, ge=1000, le=32000, description="Max context tokens"
    )
    max_response_tokens: int = Field(
        default=4096, ge=100, le=8192, description="Max response tokens"
    )
    include_sources: bool = Field(default=True, description="Include sources by default")
    min_relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum relevance")
    max_snippets: int = Field(default=20, ge=1, le=50, description="Max snippets")
    timeout: float = Field(default=60.0, ge=5.0, le=300.0, description="Timeout in seconds")


class QueryEngine:
    """Main query engine for code understanding.

    This class orchestrates the full query pipeline:
    1. Route the query to determine type
    2. Build relevant code context
    3. Call LLM with specialized prompt
    4. Format and return response
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        config: QueryEngineConfig | None = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        """Initialize the query engine.

        Args:
            llm_client: LLM client for generating responses.
            config: Engine configuration.
            context_builder: Context builder for assembling code context.
        """
        self.config = config or QueryEngineConfig()
        self._llm_client = llm_client
        self._router = QueryRouter()
        self._context_builder = context_builder or ContextBuilder(
            config=ContextConfig(
                max_tokens=self.config.max_context_tokens,
                min_relevance=self.config.min_relevance,
                max_snippets=self.config.max_snippets,
            )
        )
        self._logger = logger.bind(component="query_engine")

    async def query(
        self,
        request: QueryRequest,
        search_results: list[dict[str, Any]] | None = None,
    ) -> QueryResponse:
        """Execute a query and return the response.

        Args:
            request: The query request.
            search_results: Optional pre-fetched search results.

        Returns:
            Query response with answer and sources.
        """
        request_id = str(uuid4())
        start_time = time.time()

        self._logger.info(
            "query_started",
            request_id=request_id,
            question_preview=request.question[:50],
            repo_id=request.repo_id,
        )

        try:
            # Route the query
            route_result = self._router.route(request.question)
            query_type = route_result.query_type

            # Get the handler
            handler = get_handler(query_type)

            # Build handler context
            handler_ctx = HandlerContext(
                question=request.question,
                repo_id=request.repo_id,
                extracted_entities=route_result.extracted_entities,
                file_context=request.file_context,
            )

            # Build code context from search results
            code_context = None
            if search_results:
                code_context = self._context_builder.build_from_search_results(
                    results=search_results,
                    query=handler.build_context_query(handler_ctx),
                    repo_id=request.repo_id,
                )
                handler_ctx = HandlerContext(
                    question=request.question,
                    repo_id=request.repo_id,
                    code_context=code_context,
                    extracted_entities=route_result.extracted_entities,
                    file_context=request.file_context,
                )

            # Enhance the prompt
            enhanced_question = handler.enhance_prompt(handler_ctx)

            # Get LLM response
            llm_response = await self._get_llm_response(
                question=enhanced_question,
                context=code_context,
                query_type=query_type,
                max_tokens=request.max_tokens,
            )

            # Process the response
            result = handler.process_response(llm_response.content, handler_ctx)

            # Calculate timing
            processing_time = (time.time() - start_time) * 1000

            self._logger.info(
                "query_completed",
                request_id=request_id,
                query_type=query_type.value,
                processing_time_ms=processing_time,
                tokens_used=llm_response.input_tokens + llm_response.output_tokens,
            )

            return QueryResponse(
                request_id=request_id,
                status=QueryStatus.COMPLETED,
                query_type=query_type.value,
                result=result,
                processing_time_ms=processing_time,
                tokens_used=llm_response.input_tokens + llm_response.output_tokens,
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._logger.error(
                "query_failed",
                request_id=request_id,
                error=str(e),
                processing_time_ms=processing_time,
            )

            return QueryResponse(
                request_id=request_id,
                status=QueryStatus.FAILED,
                query_type="unknown",
                error=str(e),
                processing_time_ms=processing_time,
            )

    async def query_stream(
        self,
        request: QueryRequest,
        search_results: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Execute a query with streaming response.

        Args:
            request: The query request.
            search_results: Optional pre-fetched search results.

        Yields:
            Response text chunks.
        """
        # Route the query
        route_result = self._router.route(request.question)
        query_type = route_result.query_type

        # Get the handler
        handler = get_handler(query_type)

        # Build handler context
        handler_ctx = HandlerContext(
            question=request.question,
            repo_id=request.repo_id,
            extracted_entities=route_result.extracted_entities,
            file_context=request.file_context,
        )

        # Build code context
        code_context = None
        if search_results:
            code_context = self._context_builder.build_from_search_results(
                results=search_results,
                query=handler.build_context_query(handler_ctx),
                repo_id=request.repo_id,
            )

        # Enhance the prompt
        enhanced_question = handler.enhance_prompt(handler_ctx)

        # Stream LLM response
        if self._llm_client is None:
            raise ValueError("LLM client not configured for streaming")

        template = get_prompt_template(query_type)
        context_str = code_context.format() if code_context else ""

        system = template.system_prompt
        user = template.format_user_message(
            question=enhanced_question,
            context=context_str,
        )

        async for chunk in self._llm_client.stream(system, user):
            yield chunk

    async def _get_llm_response(
        self,
        question: str,
        context: CodeContext | None,
        query_type: Any,
        max_tokens: int,
    ) -> LLMResponse:
        """Get LLM response for a query.

        Args:
            question: The enhanced question.
            context: Code context.
            query_type: Query type.
            max_tokens: Maximum response tokens.

        Returns:
            LLM response.
        """
        if self._llm_client is None:
            # Return a mock response for testing
            return LLMResponse(
                content="This is a placeholder response. Configure an LLM client for real responses.",
                model="mock",
                input_tokens=0,
                output_tokens=0,
            )

        response = await self._llm_client.query(
            question=question,
            context=context,
            query_type=query_type,
            stream=False,
        )

        # Handle the response type
        if isinstance(response, LLMResponse):
            return response
        else:
            # Shouldn't happen with stream=False, but handle it
            raise ValueError("Unexpected response type from LLM client")

    def route(self, question: str) -> Any:
        """Route a query without executing it.

        Useful for testing and inspection.

        Args:
            question: The question to route.

        Returns:
            Routing result.
        """
        return self._router.route(question)

    def get_handler(self, query_type: Any) -> BaseHandler:
        """Get a handler for a query type.

        Args:
            query_type: The query type.

        Returns:
            Handler instance.
        """
        return get_handler(query_type)

    async def build_context(
        self,
        question: str,
        repo_id: str,
        search_results: list[dict[str, Any]],
    ) -> CodeContext:
        """Build code context from search results.

        Args:
            question: The question for context.
            repo_id: Repository identifier.
            search_results: Search results to build from.

        Returns:
            Assembled code context.
        """
        return self._context_builder.build_from_search_results(
            results=search_results,
            query=question,
            repo_id=repo_id,
        )


class MockQueryEngine(QueryEngine):
    """Mock query engine for testing.

    Uses mock LLM client and returns predictable responses.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        config: QueryEngineConfig | None = None,
    ) -> None:
        """Initialize mock query engine.

        Args:
            responses: Predefined responses to return.
            config: Engine configuration.
        """
        mock_client = MockClaudeClient(responses=responses)
        super().__init__(llm_client=mock_client, config=config)

    async def query(
        self,
        request: QueryRequest,
        search_results: list[dict[str, Any]] | None = None,
    ) -> QueryResponse:
        """Execute a mock query.

        Returns predictable responses for testing.
        """
        return await super().query(request, search_results)
